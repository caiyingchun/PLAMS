Tuning the range separation
---------------------------

In this example we optimize the value of *gamma* parameter for long-range corrected XC functional (in our case: LCY-PBE) in ADF.
Long-range corrected XC functionals can be used in ADF with XCfun (see `ADF manual  <../../ADF/Input/Density_Functional.html#range-separated-hybrids>`_).

The optimal range separation parameter *gamma* yields the HOMO energy equal to the ionization potential (IP).
Given a molecular system, we simultaneously minimize the difference between HOMO and IP for that system (N) and its anion (A) (system with one more electron). We define the J function as:

.. math::

   J = \sqrt{N^2+A^2}

and find the value of *gamma* (within a certain range) which minimizes J.

We define a new job type ``GammaJob`` by extending |MultiJob|.
The goal of ``GammaJob`` is to calculate the J function for one fixed value of *gamma*
To do that we need to perform 3 different single point calculations: 1 for the given system (let's call it S0), 1 for the system with one more electron (S-) and 1 for the system with one less electron (S+).
S+ calculation is needed to find the ionization potential of S0.

The constructor (``__init__``) of ``GammaJob`` accepts several new arguments and simply stores them.
These new arguments define: the value of *gamma*, the |Molecule| together with its initial charge, and the values of spin for S-, S0 and S+ (as a tuple of length 3).
Then the |prerun| method is used to create three children jobs with different values of total charge and spin multiplicity.
A dedicated |Results| subclass features a simple method for extracting the value of J based on results on three children jobs::

    class GammaResults(Results):

        @staticmethod
        def get_difference(job, jobplus):
            """Calculate the difference between HOMO and IP.
            *jobplus* should be the counterpart of *job* with one less electron."""
            homo = job.results.get_properties()['HOMO']
            IP = jobplus.results.get_energy() - job.results.get_energy()
            return IP + homo

        def get_J(self):
            N = GammaResults.get_difference(self.job.children[1], self.job.children[2])
            A = GammaResults.get_difference(self.job.children[0], self.job.children[1])
            return (N*N + A*A)**0.5

    class GammaJob(MultiJob):
        _result_type = GammaResults

        def __init__(self, molecule, gamma, charge, spins, **kwargs):
            MultiJob.__init__(self, **kwargs)
            self.molecule = molecule
            self.charge = charge
            self.spins = spins
            self.gamma = gamma

        def prerun(self):
            charges = [self.charge-1, self.charge, self.charge+1]
            for charge, spin in zip(charges, self.spins):
                name = str(charge)
                newjob = ADFJob(name=name, molecule=self.molecule, settings=self.settings)
                newjob.settings.input.charge = '{} {}'.format(charge, spin)
                newjob.settings.input.xc.rangesep = "gamma={:f}".format(self.gamma)
                if spin != 0:
                    newjob.settings.input.unrestricted = True
                self.children.append(newjob)

Now we can treat our newly defined ``GammaJob`` as a blackbox with simple interface: input *gamma* -> run -> extract J.
The next step is to create multiple instances of ``GammaJob`` for a range of different *gammas*.
That task can be conveniently wrapped in a simple function::

    def gamma_scan(gammas, settings, molecule, name='gammascan', charge=0, spins=(1,0,1)):
        """Calculate values of J function for given range of gammas.

        Arguments:
        gammas   - list of gamma values to calculate the J function for
        settings - Settings object compatible with ADFJob
        molecule - Molecule object with the system of interest
        name     - base name of all the jobs
        charge   - base charge of the system of interest. The J function is going to be
                   calculated based on two systems: with charge, and charge-1
        spins    - values of spin polarization (see keyword CHARGE of ADF) for jobs with,
                   respectively, charge-1, charge and charge +1

        In other words, if charge=X and spins=(a,b,c) the three resulting jobs
        are going to have the following values of CHARGE keyword:

        CHARGE X-1  a
        CHARGE   X  b
        CHARGE X+1  c

        Returns a list of pairs (gamma, J) of the same length as the parameter *gammas*
        """
        jobs = [GammaJob(molecule=molecule, settings=settings, gamma=g,
                charge=charge, spins=spins, name=name+str(g)) for g in gammas]
        results = [j.run() for j in jobs]
        js = [r.get_J() for r in results]
        return list(zip(gammas, js))

(Alternatively, instead of a function, we could define a new type of |MultiJob| with the same functionality.
Such a job would create a list of ``GammaJob`` instances as its children.
The difference in that case is rather cosmetic: in case of a new job type all ``GammaJob`` data would be stored inside new job's folder, whereas with the function defined above that data ends up directly in the main working folder.
When running the gamma scan for a lot of different molecules in one script, the function approach can lead to the main working folder being somehow messy and hard to navigate.
The new job type approach would keep the data for different molecules in different subfolders of the main working folder.)

An example usage of our newly defined function::

    from numpy import arange
    config.default_jobrunner = JobRunner(parallel=True, maxjobs=8)

    s = Settings()
    s.input.basis.type = 'TZP'
    s.input.basis.core = 'None'
    s.input.xc.gga = 'PBE'
    s.input.xc.xcfun = True
    s.runscript.nproc = 1

    mol = Molecule('somemolecule.xyz')
    gammas = arange(0.4, 0.8, 0.02)

    results = gamma_scan(gammas, s, mol)

    log('gamma \t J')
    for g,j in results:
        log('{:.4f} \t {:.8f}'.format(g,j))
    log('Optimal gamma value: {:.4f}'.format(min(results,key=lambda x:x[1])[0]))

All the code presented in above snippets can be put into a single file and executed with ``plams onebigfile.py`` (or ``$ADFBIN/plams onebigfile.py`` if ``$ADFBIN`` is not in your ``$PATH``).
Alternatively, one can place the definitions (of ``GammaJob`` and ``gamma_scan`` ) in one file ``gammajob.py`` and the execution in a separate small script ``rungamma.py`` and call it with ``plams gammajob.py rungamma.py``.
