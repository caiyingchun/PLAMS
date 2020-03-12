"""Class to manipulate CP2K jobs."""
import subprocess
import shutil
from os.path import join as opj

from ...core.basejob import SingleJob
from ...core.settings import Settings
from ...core.results import Results
from ...core.errors import ResultsError
from ...mol.molecule import Molecule
from ...mol.atom import Atom

__all__ = ['Cp2kJob', 'Cp2kResults', 'Cp2kSettings2Mol']


class Cp2kResults(Results):
    """A class for CP2K results."""

    def recreate_settings(self):
        """Recreate job for |load_external|.

        If a keyword and a section with the same name appear, only the keyword is used.
        This happens e.g. when reading restart files where ``kind.symbol.potential`` is given
        as *Potential ABCD* and *&POTENTIAL .... &END POTENTIAL*.

        Limited support of sections that have different formatting like *KIND* and *COORD*.
        Check the resulting |Settings| instance if the information you want is there.
        Be careful with reading restart files, since they are automatically generated
        and not every case is handled well here.
        You should get all the information but not sure if I know about all special cases of input.
        """

        _reserved_keywords = ["KIND", "@SET", "@INCLUDE", "@IF"]
        _different_keywords = ["COORD", "VELOCITY",
                               "MASS", "FORCE"]  # blocks of information

        def input_generator(f):
            """Yield lines from input."""
            while True:
                line = f.readline()
                if not line:
                    break
                yield line

        def parse(input_iter, res_dic):
            """Get input line and create a section or key from it.
            If a section is created, input_iter.next() is used to get all the lines
            from that section.
            So input_iter should not be a string but an iterable containing a string.

            Returns False when section is completed."""
            string = next(input_iter).strip()
            l = string.split()
            # empty line
            if not string:
                return True
            # comment line:
            elif string.startswith('#'):
                return True
            # end section
            elif string.startswith('&END'):
                return False
            # special cases
            elif any(k in string for k in _reserved_keywords):
                if '@' in string:
                    l[0] = l[0].replace('@', 'AT_')
                    res_dic.update({l[0].lower(): " ".join(l[1:])})
                elif 'KIND' in string:
                    if not 'kind' in res_dic:
                        res_dic['kind'] = {}
                    res_dic['kind'][l[1].lower()] = {}
                    r = True
                    while r:
                        r = parse(input_iter, res_dic['kind'][l[1].lower()])
                return True
            elif any("&"+k == string for k in _different_keywords):
                # save the entire block as one string until &END
                l[0] = l[0].replace('&', '')
                res_dic[l[0].lower()] = {'_h': ""}
                r = True
                while r:
                    r = next(input_iter).strip()
                    if "&" in r:
                        r = False
                        break
                    res_dic[l[0].lower()]['_h'] += "\n"
                    res_dic[l[0].lower()]['_h'] += r
                return True
            # section
            elif string.startswith('&'):
                l[0] = l[0].replace('&', '')
                # if section already exists as a key, use the key
                if l[0].lower() in res_dic:
                    # fast forward to the next &END
                    r = True
                    while r:
                        r = next(input_iter).strip()
                        if r.startswith('&END'):
                            r = False
                            break
                    return True
                res_dic[l[0].lower()] = {}
                # if section has a header value
                if len(l) > 1:
                    res_dic[l[0].lower()]['_h'] = " ".join(l[1:])
                # parse content of section
                r = True
                while r:
                    r = parse(input_iter, res_dic[l[0].lower()])
                return True

            # add key and value to dict
            else:
                res_dic.update({l[0].lower(): " ".join(l[1:])})
                return True

        dic = {}
        with open(opj(self.job.path, self.job._filename('inp'))) as f:
            input_string = input_generator(f)
            while input_string:
                try:
                    parse(input_string, dic)
                except StopIteration:
                    # nasty, but at least you get partial settings
                    break

        s = Settings()
        s.input.update(dic)
        return s

    def get_runtime(self):
        """Return runtime in seconds from output."""
        from datetime import datetime
        start = " ".join(self.grep_output(
            'PROGRAM STARTED AT')[-1].split()[-2:])
        end = " ".join(self.grep_output('PROGRAM ENDED AT')[-1].split()[-2:])
        startTime = datetime.fromisoformat(start)
        endTime = datetime.fromisoformat(end)
        diff = endTime - startTime
        return diff.total_seconds()

    def _get_energy_type(self, search='Total', index=0):
        if index:
            select = index
        else:
            select = -1
        s = self.grep_output(search + ' energy:')[select].split()[-1]
        return float(s)

    def get_energy(self, index=0):
        """Return last occurence of 'Total energy:' in the output."""
        return self._get_energy_type('Total', index=index)

    def get_dispersion(self, index=0):
        """Return last occurence of 'Dispersion energy:' in the output."""
        return self._get_energy_type('Dispersion', index=index)

    def _idx_to_match(self, nTotal, idx):
        if idx is None:
            return 0
        elif idx >= 0 and idx < nTotal:
            return idx + 1
        elif idx < -nTotal or idx >= nTotal:
            raise ResultsError("Trying to select a non-existing index.")
        else:
            return nTotal + idx + 1

    def _chunks(self, l, n, skip=0):
        ret = []
        step = len(l) // n
        for i in range(0, len(l), step):
            ret.append(l[i+skip:i+step])
        return ret

    def _get_charges(self, return_spin=False, index=-1, name='Mulliken'):
        if name == 'Mulliken':
            searchBegin = "Mulliken Population Analysis"
            searchEnd = " # Total charge and spin"
            selectCharge = -2
            selectSpin = -1
        if name == 'Hirshfeld':
            searchBegin = "Hirshfeld Charges"
            searchEnd = "Total Charge"
            selectCharge = -1
            selectSpin = -2
        n = len(self.grep_output(searchBegin))
        match = self._idx_to_match(n, index)
        chunk = self.get_output_chunk(
            begin=searchBegin, end=searchEnd, match=match)
        if match == 0:
            chunk = self._chunks(chunk, n, skip=2)
        else:
            chunk = [chunk[2:]]
        charges = []
        spin = []
        for ch in chunk:
            charges.append([float(line.strip().split()[selectCharge])
                            for line in ch])
            if return_spin:
                spin.append([float(line.strip().split()[selectSpin])
                             for line in ch])
        if return_spin:
            if match == 0:
                return charges, spin
            else:
                return charges[0], spin[0]
        else:
            if match == 0:
                return charges
            else:
                return charges[0]

    def get_mulliken_charges(self, return_spin=False, index=-1):
        """Get Mulliken charges (and spin moments).

        Set ``index`` to choose the n-th occurence of the Charges in the output, e.g. to choose an optimization step.
        Set to *None* to return all as a list.
        Defaults to the last occurence.

        Returns list of charges. If ``return_spin`` is `True` returns tuple of charges and spins.
        """
        return self._get_charges(return_spin, index, 'Mulliken')

    def get_hirshfeld_charges(self, return_spin=False, index=-1):
        """Get Hirshfeld charges (and spin moments).

        Set ``index`` to choose the n-th occurence of the Charges in the output, e.g. to choose an optimization step.
        Set to *None* to return all as a list.
        Defaults to the last occurence.

        Returns list of charges. If ``return_spin`` is `True` returns tuple of charges and spins.
        """
        return self._get_charges(return_spin, index, 'Hirshfeld')

    def get_multigrid_info(self):
        """Get Information on multigrids.

        Usefull for converging cutoffs.
        Needs 'Medium' global print level.

        Returns a dict with keys 'counts' and 'cutoffs'.
        """
        dic = {'counts': [], 'cutoffs': []}

        s = self.get_output_chunk(
            begin='MULTIGRID INFO', end='total gridlevel count')[1:]
        for line in s:
            split = line.strip().split()
            dic['counts'].append(int(split[4]))
            dic['cutoffs'].append(float(split[-1]))

        return dic


class Cp2kJob(SingleJob):
    """A class representing a single computational job with `CP2K <https://www.cp2k.org/>`_.

    In addition to the arguments of |SingleJob|, |Cp2kJob| takes a ``copy`` argument.
    ``copy`` can be a list or string, containing paths to files to be copied to the jobs directory.
    This might e.g. be a molecule, further input files etc.
    """
    _result_type = Cp2kResults

    def __init__(self, copy=None, **kwargs):
        SingleJob.__init__(self, **kwargs)
        self.copy_files = copy

    def _get_ready(self):
        """Copy files to execution dir if self.copy_files is set."""
        SingleJob._get_ready(self)
        if self.copy_files:
            if not isinstance(self.copy_files, list):
                self.copy_files = [self.copy_files]
            for f in self.copy_files:
                shutil.copy(f, self.path)
        return

    def get_input(self):
        """
        Transform all contents of ``input`` branch of |Settings| into string
        with blocks, subblocks, keys and values.
        """

        _reserved_keywords = ["KIND", "AT_SET", "AT_INCLUDE", "AT_IF"]

        def parse(key, value, indent=''):
            ret = ''
            key = key.upper()
            if isinstance(value, Settings):
                if not any(k == key for k in _reserved_keywords):
                    if '_h' in value:
                        ret += '{}&{} {}\n'.format(indent, key, value['_h'])
                    else:
                        ret += '{}&{}\n'.format(indent, key)
                    for el in value:
                        if el == '_h':
                            continue
                        ret += parse(el, value[el], indent + '  ')
                    ret += '{}&END\n'.format(indent)

                elif "KIND" in key:
                    for el in value:
                        ret += '{}&{}  {}\n'.format(indent, key, el.upper())
                        for v in value[el]:
                            ret += parse(v, value[el][v], indent + '  ')
                        ret += '{}&END\n'.format(indent)

                elif "AT_SET" in key:
                    var, val = tuple(value.items())[0]
                    ret += '@SET {} {}\n'.format(var, val)

                elif "AT_IF" in key:
                    pred, branch = tuple(value.items())[0]
                    ret += '{}@IF {}\n'.format(indent, pred)
                    for k, v in branch.items():
                        ret += parse(k, v, indent + '  ')
                    ret += '{}@ENDIF\n'.format(indent)

            elif key == "AT_INCLUDE":
                ret += '@include {}\n'.format(value)

            elif isinstance(value, list):
                for el in value:
                    ret += parse(key, el, indent)

            elif value is '' or value is True:
                ret += '{}{}\n'.format(indent, key)
            else:
                ret += '{}{}  {}\n'.format(indent, key, str(value))
            return ret

        inp = ''

        if self.molecule:
            use_molecule = ('ignore_molecule' not in self.settings) or (
                self.settings.ignore_molecule == False)
            if use_molecule:
                self._parsemol()

        for item in self.settings.input:
            inp += parse(item, self.settings.input[item]) + '\n'

        return inp

    def _parsemol(self):
        # make lines shorter
        inp = self.settings.input.force_eval.subsys
        # add cell information
        nDim = len(self.molecule.lattice)
        keys = ['A', 'B', 'C']
        periodic = ['X', 'XY', 'XYZ']
        for iDim in range(0, nDim):
            inp.cell[keys[iDim]] = "{:} {:} {:}".format(
                *self.molecule.lattice[iDim])
        if nDim > 0:
            inp.cell.periodic = periodic[nDim-1]

        # get block of: symbol coords
        coord_sec = ""
        for atom in self.molecule:
            coord_sec += "\n"
            coord_sec += (" {:}"*4).format(atom.symbol, *atom.coords)
        inp.coord._h = coord_sec

    def get_runscript(self):
        """Run parallel version of Cp2k using srun."""
        # Try to run cp2k using mpirun and otherwise srun (if available)
        cp2k_command = self.settings.get("executable", "cp2k.popt")

        # Check the executable name
        available_executables = (
            # Serial single core testing and debugging
            "sdbg",
            # Serial general single core usage
            "sopt",
            # Parallel (only OpenMP), single node, multi core
            "ssmp",
            # Parallel (only MPI) multi-node testing and debugging
            "pdbg",
            # Parallel (only MPI) general usage, no threads
            "popt",
            # parallel (MPI + OpenMP) general usage, threading might improve scalability and memory usage
            "psmp")
        if not any((f"cp2k.{x}" == cp2k_command.lower() for x in available_executables)):
            msg = f"unrecognized cp2k executable: {cp2k_command}"
            raise RuntimeError(msg)

        # Try to run cp2k using mpirun and otherwise srun (if available)
        command_tuple = ('mpirun', 'srun')
        for command in command_tuple:
            try:
                subprocess.run([command, "--help"], stdout=subprocess.DEVNULL)
                ret = f"{command} {cp2k_command}"
                break
            except OSError:
                pass

        ret += ' -i {} -o {}'.format(self._filename('inp'),
                                     self._filename('out'))
        return ret

    def check(self):
        """Look for the normal termination signal in Cp2k output."""
        s = self.results.grep_output("PROGRAM STOPPED IN")
        return len(s) > 0


def Cp2kSettings2Mol(settings):
    """Return a molecule from a |Settings| instance used for a |Cp2kJob|.

    Loads coordinates from ``settings.input.force_eval.subsys.coord._h`` and
    cell information from ``settings.input.force_eval.subsys.cell``.
    """
    mol = Molecule()

    if 'force_eval' not in settings.input:
        return None
    elif 'subsys' not in settings.input.force_eval:
        return None
    elif 'coord' not in settings.input.force_eval.subsys:
        return None
    elif '_h' not in settings.input.force_eval.subsys.coord:
        return None
    coord = settings.input.force_eval.subsys.coord._h

    pbc = False
    if 'cell' in settings.input.force_eval.subsys:
        pbc = True
        cell = settings.input.force_eval.subsys.cell

    split = coord.strip().split('\n')
    for line in split:
        lineSplit = line.split()
        try:
            lineSplit[1:4] = [float(x) for x in lineSplit[1:4]]
        except ValueError:
            # not an atom entry
            continue
        mol.add_atom(Atom(symbol=lineSplit[0], coords=tuple(lineSplit[1:4])))

    if pbc:
        vec = []
        keys = ['a', 'b', 'c']
        for key in keys:
            if key not in cell:
                break
            try:
                vec.append(tuple([float(x) for x in cell[key].split()]))
            except ValueError:
                break
        mol.lattice = vec
    return mol
