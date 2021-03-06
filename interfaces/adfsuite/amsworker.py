import os
import shutil
import subprocess
import threading
import queue
import struct
import weakref
import tempfile
import functools
import numpy as np
import collections
from typing import *

try:
    import ubjson
    __all__ = ['AMSWorker', 'AMSWorkerResults', 'AMSWorkerError', 'AMSWorkerPool']
except ImportError:
    __all__ = []

from ...mol.molecule import Molecule
from ...core.settings import Settings
from ...core.errors import PlamsError, JobError, ResultsError
from ...tools.units import Units
from ...core.functions import config, log
from .ams import AMSJob
from .amspipeerror import *


def _restrict(func):
    """Decorator that wraps methods of |AMSWorkerResults| instances.

    This is used to replicate the behaviour of the full |AMSResults| object: Access to the values in an |AMSWorkerResults| instance will first check if the calculation leading to the results finish correctly and raise a |ResultsError| error exception if this is not the case. This behaviour can be modified with the ``config.ignore_failure`` setting.
    """
    @functools.wraps(func)
    def guardian(self, *args, **kwargs):
        if self.ok():
            return func(self, *args, **kwargs)
        else:
            if config.ignore_failure:
                log('WARNING: Trying to obtain results of a failed calculation {}'.format(self.name), 3)
                try:
                    ret = func(self, *args, **kwargs)
                except:
                    log('Obtaining results of {} failed. Returned value is None'.format(self.name), 3)
                    return None
                log('Obtaining results of {} successful. However, no guarantee that they make sense'.format(self.name), 3)
                return ret
            else:
                raise ResultsError('Using Results obtained from a failed calculation')
    return guardian



class AMSWorkerResults:
    """A specialized class encapsulating the results from calls to an |AMSWorker|.

    .. technical::

        AMSWorkerResults is *not* a subclass of |Results| or |AMSResults|. It does however implement some commonly used methods of the |AMSResults| class, so that results calculated by |AMSJob| and |AMSWorker| can be accessed in a uniform way.
    """

    def __init__(self, name, molecule, results, error=None):
        self._name           = name
        self._input_molecule = molecule
        self.error           = error
        self._results        = results
        if self._results is not None and 'xyzAtoms' in self._results:
            self._main_molecule = self._input_molecule.copy()
            self._main_molecule.from_array(self._results.pop('xyzAtoms') * Units.conversion_ratio('au', 'Angstrom'))
            if 'latticeVectors' in self._results:
                self._main_molecule.lattice = [ tuple(v) for v in self._results.pop('latticeVectors') * Units.conversion_ratio('au', 'Angstrom') ]
        else:
            self._main_molecule = self._input_molecule

    @property
    def name(self):
        """The name of a calculation.

        That is the name that was passed into the |AMSWorker| method when this |AMSWorkerResults| object was created. I can not be changed after the |AMSWorkerResults| instance has been created.
        """
        return self._name
    @name.setter
    def name(self, _):
        raise ResultsError('The name attribute of AMSWorkerResults may not be changed.')

    def ok(self):
        """Check if the calculation was successful. If not, the ``error`` attribute contains a corresponding exception.

        Users should check if the calculation was successful before using the other methods of the |AMSWorkerResults| instance, as using them might raise a |ResultsError| exception otherwise.
        """
        return self.error is None

    def get_errormsg(self):
        if self.ok():
            return None
        else:
            lines = str(self.error).splitlines()
            if lines:
                for line in reversed(lines):
                    if 'ERROR: ' in line:
                        return line.partition('ERROR: ')[2]
                return lines[-1]
            else:
                return 'Could not determine error message. Please check the error.stdout and error.stderr manually.'

    @_restrict
    def get_energy(self, unit='au'):
        """Return the total energy, expressed in *unit*.
        """
        return self._results["energy"] * Units.conversion_ratio('au', unit)

    @_restrict
    def get_gradients(self, energy_unit='au', dist_unit='au'):
        """Return the nuclear gradients of the total energy, expressed in *energy_unit* / *dist_unit*.
        """
        return self._results["gradients"] * Units.conversion_ratio('au', energy_unit) / Units.conversion_ratio('au', dist_unit)

    @_restrict
    def get_stresstensor(self):
        """Return the clamped-ion stress tensor, expressed in atomic units.
        """
        return self._results["stressTensor"]

    @_restrict
    def get_hessian(self):
        """Return the Hessian matrix, i.e. the second derivative of the total energy with respect to the nuclear coordinates, expressed in atomic units.
        """
        return self._results["hessian"]

    @_restrict
    def get_elastictensor(self):
        """Return the elastic tensor, expressed in atomic units.
        """
        return self._results["elasticTensor"]

    @_restrict
    def get_charges(self):
        """Return the atomic charges, expressed in atomic units.
        """
        return self._results["charges"]

    @_restrict
    def get_dipolemoment(self):
        """Return the electric dipole moment, expressed in atomic units.
        """
        return self._results["dipoleMoment"]

    @_restrict
    def get_dipolegradients(self):
        """Return the nuclear gradients of the electric dipole moment, expressed in atomic units. This is a (3*numAtoms x 3) matrix.
        """
        return self._results["dipoleGradients"]

    def get_input_molecule(self):
        """Return a |Molecule| instance with the coordinates passed into the |AMSWorker|.

        Note that this method may also be used if the calculation producing this |AMSWorkerResults| object has failed, i.e. :meth:`ok` is ``False``."""
        return self._input_molecule

    @_restrict
    def get_main_molecule(self):
        """Return a |Molecule| instance with the final coordinates.
        """
        return self._main_molecule



class AMSWorkerError(PlamsError):
    """Error related to an AMSWorker process.

    The output from the failed worker process is stored in the ``stdout`` and ``stderr`` attributes.
    """

    def __init__(self, *args):
        super().__init__(*args)
        self.stdout = None
        self.stderr = None

    def __str__(self):
        msg = super().__str__()
        if self.stderr is not None:
            return "".join([msg, "\n"] + self.stderr)
        else:
            return msg

    def get_errormsg(self):
        lines = str(self).splitlines()
        if lines:
            for line in reversed(lines):
                if 'ERROR: ' in line:
                    return line.partition('ERROR: ')[2]
            return lines[-1]
        else:
            return 'Could not determine error message. Please check the error.stdout and error.stderr manually.'



_arg2setting = {}

for x in ("prev_results", "quiet"):
    _arg2setting[x] = ('amsworker', x)

for x in ("gradients", "stresstensor", "hessian", "elastictensor", "charges", "dipolemoment", "dipolegradients"):
    _arg2setting[x] = ('input', 'ams', 'properties', x)

for x in ("coordinatetype", "optimizelattice", "maxiterations", "pretendconverged"):
    _arg2setting[x] = ('input', 'ams', 'geometryoptimization', x)

for x in ("convenergy", "convgradients", "convstep", "convstressenergyperatom"):
    _arg2setting[x] = ('input', 'ams', 'geometryoptimization', 'convergence', x[4:])

_arg2setting['task'] = ('input', 'ams', 'task')
_arg2setting['usesymmetry'] = ('input', 'ams', 'usesymmetry')
_arg2setting['method'] = ('input', 'ams', 'geometryoptimization', 'method')

_setting2arg = {s: a for a, s in _arg2setting.items()}


class AMSWorker:
    """A class representing a running instance of the AMS driver as a worker process.

    Users need to supply a |Settings| instance representing the input of the AMS driver process (see :ref:`AMS_preparing_input`), but **not including** the ``Task`` keyword in the input (the ``input.ams.Task`` key in the |Settings| instance). The |Settings| instance should also not contain a system specification in the ``input.ams.System`` block. Often the settings of the AMS driver in worker mode will come down to just the engine block.

    The AMS driver will then start up as a worker, communicating with PLAMS via named pipes created in a temporary directory (determined by the *workerdir_root* and *workerdir_prefix* arguments). This temporary directory might also contain temporary files used by the worker process. Note that while an |AMSWorker| instance exists, the associated worker process can be assumed to be running and ready: If it crashes for some reason it is automatically restarted.

    The recommended way to start an |AMSWorker| is as a context manager:

    .. code-block:: python

        with AMSWorker(settings) as worker:
            results = worker.SinglePoint('my_calculation', molecule)
        # clean up happens automatically when leaving the block

    If it is not possible to use the |AMSWorker| as a context manager, cleanup should be manually triggered by calling the :meth:`stop` method.
    """

    def __init__(self, settings, workerdir_root=None, workerdir_prefix='amsworker', use_restart_cache=True):

        self.PyProtVersion = 1
        self.timeout = 2.0
        self.use_restart_cache = use_restart_cache

        # We'll initialize these communication related variables to None for now.
        # They will be overwritten when we actually start things up, but we do
        # not want them to be undefined for now, just in case of errors ...
        self.proc      = None
        self.callpipe  = None
        self.replypipe = None

        self.restart_cache = set()
        self.restart_cache_deleted = set()

        # Make a copy of the Settings instance so we do not modify the outside world and fix the task to be "Pipe".
        self.settings = settings.copy()
        self.settings.input.ams.task = 'pipe'

        # Check if the settings we have are actually suitable for a PipeWorker.
        # They should not contain the Task keyword and no System block.
        if 'ams' in settings.input:
            task = settings.input.ams.find_case('task')
            if task in settings.input.ams:
                raise JobError('Settings for AMSWorker should not contain a Task')
            system = settings.input.ams.find_case('system')
            if system in settings.input.ams:
                raise JobError('Settings for AMSWorker should not contain a System block')

        # Create the directory in which we will run the worker.
        self.workerdir = tempfile.mkdtemp(dir=workerdir_root, prefix=workerdir_prefix+'_')
        weakref.finalize(self, shutil.rmtree, self.workerdir)

        # Start the worker process.
        self._start_subprocess()


    def _start_subprocess(self):

        # We will use the standard PLAMS AMSJob class to prepare our input and runscript.
        amsjob = AMSJob(name='amsworker', settings=self.settings)
        with open(os.path.join(self.workerdir, 'amsworker.in'), 'w') as input_file:
            input_file.write(amsjob.get_input())
        with open(os.path.join(self.workerdir, 'amsworker.run'), 'w') as runscript:
            runscript.write(amsjob.get_runscript())
        del amsjob

        # Create the named pipes for communication.
        for filename in ["call_pipe", "reply_pipe"]:
            os.mkfifo(os.path.join(self.workerdir, filename))

        # Launch the worker process
        with open(os.path.join(self.workerdir, 'amsworker.in'), 'r') as amsinput, \
             open(os.path.join(self.workerdir, 'ams.out'), 'w') as amsoutput, \
             open(os.path.join(self.workerdir, 'ams.err'), 'w') as amserror:
            self.proc = subprocess.Popen(['sh', 'amsworker.run'], cwd=self.workerdir, stdout=amsoutput, stdin=amsinput, stderr=amserror)

        # Start a dedicated watcher thread to rescue us in case the worker never opens its end of the pipes.
        self._stop_watcher = threading.Event()
        self._watcher = threading.Thread(target=self._startup_watcher, args=[self.workerdir], daemon=True)
        try:
            self._watcher.start()

            # These two will block until either the worker is ready or the watcher steps in.
            self.callpipe  = open(os.path.join(self.workerdir, 'call_pipe'), 'wb')
            self.replypipe = open(os.path.join(self.workerdir, 'reply_pipe'), 'rb')
        finally:
            # Both open()s are either done or have failed, we don't need the watcher thread anymore.
            self._stop_watcher.set()
            self._watcher.join()

        # Raise a nice error message if the worker failed to start. Otherwise, we'd get
        # a less descriptive error from the call to Hello below.
        try:
            if not self._check_process():
                raise AMSWorkerError('AMSWorker process did not start up correctly')

            # Now everything should be ready. Let's try saying Hello via the pipes ...
            self._call("Hello", {"version": self.PyProtVersion})
        except AMSWorkerError as exc:
            exc.stdout, exc.stderr = self.stop()
            raise


    def _startup_watcher(self, workerdir):
        while not self._stop_watcher.is_set():
            try:
                self.proc.wait(timeout=0.01)
                # self.proc has died and won't open its end of the pipes ...
                if not self._stop_watcher.is_set():
                    # ... but the main thread is still expecting someone to do it.
                    # Let's do it ourselves to unblock our main thread.
                    with open(os.path.join(workerdir, 'call_pipe'), 'rb'), \
                         open(os.path.join(workerdir, 'reply_pipe'), 'wb'):
                        # Nothing to do here, just close the pipes again.
                        pass
                return
            except subprocess.TimeoutExpired:
                # self.proc is still alive.
                pass


    def __enter__(self):
        return self


    def stop(self):
        """Stops the worker process and removes its working directory.

        This method should be called when the |AMSWorker| instance is not used as a context manager and the instance is no longer needed. Otherwise proper cleanup is not guaranteed to happen, the worker process might be left running and files might be left on disk.
        """

        stdout = None
        stderr = None

        # Stop the worker process.
        if self.proc is not None:
            if self._check_process():
                try:
                    self._call("Exit")
                    self.proc.wait(timeout=self.timeout)
                except (AMSWorkerError, subprocess.TimeoutExpired):
                    self.proc.kill()
                    self.proc.wait()
            self.proc = None
            with open(os.path.join(self.workerdir, 'ams.out'), 'r') as amsoutput:
                stdout = amsoutput.readlines()
            with open(os.path.join(self.workerdir, 'ams.err'), 'r') as amserror:
                stderr = amserror.readlines()

        # At this point the worker is down. We definitely don't have anything in the restart cache anymore ...
        self.restart_cache.clear()
        self.restart_cache_deleted.clear()

        # Tear down the pipes.
        if self.callpipe is not None:
            if not self.callpipe.closed:
                try:
                    self.callpipe.close()
                except BrokenPipeError:
                    pass
            self.callpipe = None
        if self.replypipe is not None:
            if not self.replypipe.closed:
                try:
                    self.replypipe.close()
                except BrokenPipeError:
                    pass
            self.readpipe = None

        # Remove the contents of the worker directory.
        for filename in os.listdir(self.workerdir):
            file_path = os.path.join(self.workerdir, filename)
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            else:
                os.unlink(file_path)

        return (stdout, stderr)


    def __exit__(self, *args):
        self.stop()


    def _delete_from_restart_cache(self, name):
        if name in self.restart_cache:
            self.restart_cache.remove(name)
            self.restart_cache_deleted.add(name)


    def _prune_restart_cache(self):
        for name in list(self.restart_cache_deleted):
            self._call("DeleteResults", {"title": name})
        self.restart_cache_deleted.clear()


    @staticmethod
    def supports(s:Settings) -> bool:
        '''
        Check if a |Settings| object is supported by |AMSWorker|.
        '''

        try:
            args = AMSWorker._settings_to_args(s)
            return True
        except NotImplementedError:
            return False


    @staticmethod
    def _settings_to_args(s:Settings) -> Tuple[ str, Dict ]:
        '''
        Return a `tuple(TASK, **request_kwargs)` corresponding to a given settings object.

        Raises NotImplementedError if unsupported features are encountered.
        '''

        args = {}
        for key, val in s.flatten().items():
            kl = tuple(x.lower() for x in key)
            try:
                args[_setting2arg[kl]] = val
            except KeyError:
                raise NotImplementedError("Unexpected key {}".format(".".join(key)))

        if "task" not in args:
            raise NotImplementedError("No Settings.input.ams.task found")
        elif args["task"].lower() not in ("singlepoint", "geometryoptimization"):
            raise NotImplementedError("Unexpected task {}".format(args["task"]))

        return args


    @staticmethod
    def _args_to_settings(**kwargs) -> Settings:
        s = Settings()

        for key, val in kwargs.items():
            s.set_nested(_arg2setting[key], val)

        return s


    def evaluate(self, name, molecule, settings):
        args = AMSWorker._settings_to_args(settings)

        return self._solve(name, molecule, **args)


    def _solve(self, name, molecule, task, prev_results=None, quiet=True,
               gradients=False, stresstensor=False, hessian=False, elastictensor=False,
               charges=False, dipolemoment=False, dipolegradients=False,
               method=None, coordinatetype=None, usesymmetry=None, optimizelattice=False,
               maxiterations=None, pretendconverged=None,
               convenergy=None, convgradients=None, convstep=None, convstressenergyperatom=None):

        if self.use_restart_cache and name in self.restart_cache:
            raise JobError(f'Name "{name}" is already associated with results from the restart cache.')

        try:

            # This is a good opportunity to let the worker process know about all the results we no longer need ...
            self._prune_restart_cache()

            chemicalSystem = {}
            chemicalSystem['atomSymbols'] = np.asarray([atom.symbol for atom in molecule])
            chemicalSystem['coords'] = molecule.as_array() * Units.conversion_ratio('Angstrom','Bohr')
            if 'charge' in molecule.properties:
                chemicalSystem['totalCharge'] = float(molecule.properties.charge)
            else:
                chemicalSystem['totalCharge'] = 0.0
            self._call("SetSystem", chemicalSystem)
            if molecule.lattice:
                cell = np.asarray(molecule.lattice) * Units.conversion_ratio('Angstrom','Bohr')
                self._call("SetLattice", {"vectors": cell})
            else:
                self._call("SetLattice", {})

            args = {
                "request": { "title": str(name) },
                "keepResults": self.use_restart_cache,
            }
            if quiet: args["request"]["quiet"] = True
            if gradients: args["request"]["gradients"] = True
            if stresstensor: args["request"]["stressTensor"] = True
            if hessian: args["request"]["hessian"] = True
            if elastictensor: args["request"]["elasticTensor"] = True
            if charges: args["request"]["charges"] = True
            if dipolemoment: args["request"]["dipoleMoment"] = True
            if dipolegradients: args["request"]["dipoleGradients"] = True
            if self.use_restart_cache and prev_results is not None and prev_results.name in self.restart_cache:
                args["prevTitle"] = prev_results.name

            if task == 'geometryoptimization':
                if method is not None: args["method"] = method
                if coordinatetype is not None: args["coordinateType"] = coordinatetype
                if usesymmetry is not None: args["useSymmetry"] = usesymmetry
                if optimizelattice: args["optimizeLattice"] = True
                if maxiterations is not None: args["maxIterations"] = maxiterations
                if pretendconverged: args["pretendConverged"] = True
                if convenergy is not None: args["convEnergy"] = convenergy
                if convgradients is not None: args["convGradients"] = convgradients
                if convstep is not None: args["convStep"] = convstep
                if convstressenergyperatom is not None: args["convStressEnergyPerAtom"] = convstressenergyperatom
                results = self._call("Optimize", args)
            else:
                results = self._call("Solve", args)

            results = self._unflatten_arrays(results[0]['results'])
            results = AMSWorkerResults(name, molecule, results)

            if self.use_restart_cache:
                self.restart_cache.add(name)
                weakref.finalize(results, self._delete_from_restart_cache, name)

            return results

        except AMSPipeRuntimeError as exc:
            return AMSWorkerResults(name, molecule, None, exc)
        except AMSWorkerError as exc:
            # Something went wrong. Our worker process might also be down.
            # Let's reset everything to be safe ...
            exc.stdout, exc.stderr = self.stop()
            self._start_subprocess()
            # ... and return an AMSWorkerResults object indicating our failure.
            return AMSWorkerResults(name, molecule, None, exc)


    def SinglePoint(self, name, molecule, prev_results=None, quiet=True,
                    gradients=False, stresstensor=False, hessian=False, elastictensor=False,
                    charges=False, dipolemoment=False, dipolegradients=False):
        """Performs a single point calculation on the geometry given by the |Molecule| instance *molecule* and returns an instance of |AMSWorkerResults| containing the results.

        Every calculation should be given a *name*. Note that the name **must be unique** for this |AMSWorker| instance: One should not attempt to reuse calculation names with a given instance of |AMSWorker|.

        By default only the total energy is calculated but additional properties can be requested using the corresponding keyword arguments:

        - *gradients*: Calculate the nuclear gradients of the total energy.
        - *stresstensor*: Calculate the clamped-ion stress tensor. This should only be requested for periodic systems.
        - *hessian*: Calculate the Hessian matrix, i.e. the second derivative of the total energy with respect to the nuclear coordinates.
        - *elastictensor*: Calculate the elastic tensor. This should only be requested for periodic systems.
        - *charges*: Calculate atomic charges.
        - *dipolemoment*: Calculate the electric dipole moment. This should only be requested for non-periodic systems.
        - *dipolemoment*: Calculate the nuclear gradients of the electric dipole moment. This should only be requested for non-periodic systems.

        Users can pass an instance of a previously obtained |AMSWorkerResults| as the *prev_results* keyword argument. This can trigger a restart from previous results in the worker process, the details of which depend on the used computational engine: For example, a DFT based engine might restart from the electronic density obtained in an earlier calculation on a similar geometry. This is often useful to speed up series of sequentially dependent calculations:

        .. code-block:: python

            mol = Molecule('some/system.xyz')
            with AMSWorker(sett) as worker:
                last_results = None
                do i in range(num_steps):
                    results = worker.SinglePoint(f'step{i}', mol, prev_results=last_results, gradients=True)
                    # modify the geometry of mol using results.get_gradients()
                    last_results = results

        Note that the restarting is disabled if the |AMSWorker| instance was created with ``use_restart_cache=False``. It is still permitted to pass previous |AMSResults| instances as the *prev_results* argument, but no restarting will happen.

        The *quiet* keyword can be used to obtain more output from the worker process. Note that the output of the worker process is not printed to the standard output but instead ends up in the ``ams.out`` file in the temporary working directory of the |AMSWorker| instance. This is mainly useful for debugging.
        """
        args = locals()
        del args['self']
        del args['name']
        del args['molecule']
        s = self._args_to_settings(**args)
        s.input.ams.task = 'singlepoint'
        return self.evaluate(name, molecule, s)


    def GeometryOptimization(self, name, molecule, prev_results=None, quiet=True,
                             gradients=False, stresstensor=False, hessian=False, elastictensor=False,
                             charges=False, dipolemoment=False, dipolegradients=False,
                             method=None, coordinatetype=None, usesymmetry=None, optimizelattice=False,
                             maxiterations=None, pretendconverged=None,
                             convenergy=None, convgradients=None, convstep=None, convstressenergyperatom=None):
        """Performs a geometry optimization on the |Molecule| instance *molecule* and returns an instance of |AMSWorkerResults| containing the results from the optimized geometry.

        The geometry optimizer can be controlled using the following keyword arguments:

        - *method*: String identifier of a particular optimization algorithm.
        - *coordinatetype*: Select a particular kind of optimization coordinates.
        - *usesymmetry*: Enable the use of symmetry when applicable.
        - *optimizelattice*: Optimize the lattice vectors together with atomic positions.
        - *maxiterations*: Maximum number of iterations allowed.
        - *pretendconverged*: If set to true, non converged geometry optimizations will be considered successful.
        - *convenergy*: Convergence criterion for the energy (in Hartree).
        - *convgradients*: Convergence criterion for the gradients (in Hartree/Bohr).
        - *convstep*: Convergence criterion for displacements (in Bohr).
        - *convstressenergyperatom*: Convergence criterion for the stress energy per atom (in Hartree).
        """
        args = locals()
        del args['self']
        del args['name']
        del args['molecule']
        s = self._args_to_settings(**args)
        s.input.ams.task = 'geometryoptimization'
        return self.evaluate(name, molecule, s)


    def ParseInput(self, program_name, text_input):
        try:
            reply = self._call("ParseInput", {"programName": program_name, "textInput": text_input})
            json_input = reply[0]['parsedInput']['jsonInput']
            return json_input
        except AMSWorkerError as exc:
            # This failed badly, also the worker is likely down. Let's grab some info, restart it ...
            exc.stdout, exc.stderr = self.stop()
            self._start_subprocess()
            # ... and then reraise the exception for the caller.
            raise


    def _check_process(self) :
        if self.proc is not None:
            status = self.proc.poll()
            return status is None
        else:
            return False


    def _flatten_arrays(self, d):
        out = {}
        for key, val in d.items():
            if (isinstance(val, collections.abc.Sequence) or isinstance(val, np.ndarray)) and not isinstance(val, str):
                array = np.asarray(val)
                out[key] = array.flatten()
                out[key + "_dim_"] = array.shape[::-1]
            elif isinstance(val, collections.abc.Mapping):
                out[key] = self._flatten_arrays(val)
            else:
                out[key] = val
        return out


    def _unflatten_arrays(self, d):
        out = {}
        for key, val in d.items():
            if key + "_dim_" in d:
                out[key] = np.asarray(val).reshape(d[key + "_dim_"][::-1])
            elif key.endswith("_dim_"):
                pass
            elif isinstance(val, collections.abc.Mapping):
                out[key] = self._unflatten_arrays(val)
            else:
                out[key] = val
        return out

    def _read_exactly(self, pipe, n):
        buf = pipe.read(n)
        if len(buf) == n:
            return buf
        else:
            raise EOFError("Message truncated")

    def _call(self, method, args={}):
        msg = ubjson.dumpb({method: self._flatten_arrays(args)})
        msglen = struct.pack("=i", len(msg))
        try:
            self.callpipe.write(msglen + msg)
            if method.startswith("Set"):
                return None
            self.callpipe.flush()
        except BrokenPipeError as exc:
            raise AMSWorkerError('Error while sending a message') from exc
        if method == "Exit":
            return None

        results = []
        while True:
            try:
                msgbuf = self._read_exactly(self.replypipe, 4)
                msglen = struct.unpack("=i", msgbuf)[0]
                msgbuf = self._read_exactly(self.replypipe, msglen)
            except EOFError as exc:
                raise AMSWorkerError("Error while trying to read a reply") from exc

            try:
                msg = ubjson.loadb(msgbuf)
            except Exception as exc:
                raise AMSWorkerError("Error while decoding a reply") from exc

            if "return" in msg:
                ret = msg["return"]
                if ret["status"] == 0:
                    return results
                else:
                    raise AMSPipeError.from_message(ret)
            else:
                results.append(msg)



class AMSWorkerPool:
    """A class representing a pool of AMS worker processes.

    All workers of the pool are initialized with the same |Settings| instance, see the |AMSWorker| constructor for details.

    The number of spawned workers is determined by the *num_workers* argument. For optimal performance on many small jobs it is recommended to spawn a number of workers equal to the number of physical CPU cores of the machine the calculation is running on, and to let every worker instance run serially:

    .. code-block:: python

        import psutil

        molecules = read_molecules('folder/with/xyz/files')

        sett = Settings()
        # ... more settings ...
        sett.runscript.nproc = 1 # <-- every worker itself is serial (aka export NSCM=1)

        with AMSWorkerPool(sett, psutil.cpu_count(logical=False)) as pool:
            results = pool.SinglePoints([ (name, molecules[name]) for name in sorted(molecules) ])

    As with the underlying |AMSWorker| class, the location of the temporary directories can be changed with the *workerdir_root* and *workerdir_prefix* arguments.

    It is recommended to use the |AMSWorkerPool| as a context manager in order to ensure that cleanup happens automatically. If it is not possible to use the |AMSWorkerPool| as a context manager, cleanup should be manually triggered by calling the :meth:`stop` method.

    """

    def __init__(self, settings, num_workers, workerdir_root=None, workerdir_prefix='awp'):

        self.workers = num_workers * [None]

        threads = [ threading.Thread(target=AMSWorkerPool._spawn_worker, args=(self.workers, settings, i, workerdir_root, workerdir_prefix))
                    for i in range(num_workers) ]
        for t in threads: t.start()
        for t in threads: t.join()
        if None in self.workers:
            raise PlamsError('Some AMSWorkers failed to start')


    def _spawn_worker(workers, settings, i, wdr, wdp):
        workers[i] = AMSWorker(settings, workerdir_root=wdr, workerdir_prefix=f'{wdp}_{i}', use_restart_cache=False)


    def __enter__(self):
        return self


    def evaluate(self, items):
        """Request to pool to execute calculations for all items in the iterable *items*. Returns a list of |AMSWorkerResults| objects.

        The *items* argument is expected to be an iterable of 3-tuples ``(name, molecule, settings)``, which are passed on to the the :meth:`evaluate <AMSWorker.evaluate>` method of the pool's |AMSWorker| instances.
        """

        results = [None]*len(items)

        q = queue.Queue()

        threads = [ threading.Thread(target=AMSWorkerPool._execute_queue, args=(self.workers[i], q, results)) for i in range(len(self.workers)) ]
        for t in threads: t.start()

        for i, item in enumerate(items):
            if len(item) == 3:
                name, mol, settings = item
            else:
                raise JobError('AMSWorkerPool.evaluate expects a list containing only 3-tuples (name, molecule, settings).')
            q.put((i, name, mol, settings))

        for t in threads: q.put(None) # signal for the thread to end
        for t in threads: t.join()

        return results


    def SinglePoints(self, items):
        """Request to pool to execute single point calculations for all items in the iterable *items*. Returns a list of |AMSWorkerResults| objects.

        The *items* argument is expected to be an iterable of 2-tuples ``(name, molecule)`` and/or 3-tuples ``(name, molecule, kwargs)``, which are passed on to the :meth:`SinglePoint <AMSWorker.SinglePoint>` method of the pool's |AMSWorker| instances. (Here ``kwargs`` is a dictionary containing the optional keyword arguments and their values for this method.)

        As an example, the following call would do single point calculations with gradients and (only for periodic systems) stress tensors for all |Molecule| instances in the dictionary ``molecules``.

        .. code-block:: python

            results = pool.SinglePoint([ (name, molecules[name], {
                                             "gradients": True,
                                             "stresstensor": len(molecules[name].lattice) != 0
                                          }) for name in sorted(molecules) ])
        """

        evalitems = []
        for item in items:
            if len(item) == 2:
                name, mol = item
                kwargs = {}
            elif len(item) == 3:
                name, mol, kwargs = item
            else:
                raise JobError('AMSWorkerPool.SinglePoints expects a list containing only 2-tuples (name, molecule) and/or 3-tuples (name, molecule, kwargs).')
            s = AMSWorker._args_to_settings(**kwargs)
            s.input.ams.task = 'singlepoint'

            evalitems.append((name, mol, s))

        return self.evaluate(evalitems)



    def _execute_queue(worker, q, results):
        while True:
            item = q.get()
            try:
                if item is None:
                    break
                i, name, mol, settings = item
                results[i] = worker.evaluate(name, mol, settings)
            finally:
                q.task_done()


    def stop(self):
        """Stops the all worker processes and removes their working directories.

        This method should be called when the |AMSWorkerPool| instance is not used as a context manager and the instance is no longer needed. Otherwise proper cleanup is not guaranteed to happen, worker processes might be left running and files might be left on disk.
        """
        for worker in self.workers:
            worker.stop()


    def __exit__(self, *args):
        self.stop()
