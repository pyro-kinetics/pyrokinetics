.. default-role:: math
.. default-domain:: py
.. _sec-hpc-systems:

Using Pyrokinetics on HPC Systems
=================================

.. attention::

   The examples in this section were written to work on the Viking HPC cluster at the
   University of York. Your machine may be set up differently, particularly when it
   comes to partitions, scratch drives, and module names.

Pyrokinetics may be used on High-Performance Computing (HPC) systems to assist in
setting up gyrokinetics simulations. This tutorial will explain how to set up
Pyrokinetics on these machines, how to generate input files at run time, and how to
use these alongside pre-compiled gyrokinetics codes. It will also cover some advanced
techniques, such as how to manage jobs within a single ``sbatch`` call, and how to
dispatch jobs to a remote server.

.. _sec-installing-hpc:

Installation
------------

To install Pyrokinetics on an HPC system, we'll first need to log in using SSH:

.. code-block:: bash

   $ ssh username@hpc.machine.com

You may need to contact the system administrators beforehand to receive your access
credentials. Once you've logged in, you'll then need to set up a Python environment:

.. code-block:: bash

   $ module load Python  # This may vary depending on your system!
   $ mkdir ~/.local/venv
   $ python -m venv ~/.local/venv/pyrokinetics
   $ source ~/.local/venv/pyrokinetics/bin/activate

After creating a new Python environment, you may either install Pyrokinetics as a PyPI
package, or by cloning the repo:

.. code-block:: bash

   $ # Use PyPI package
   $ pip install pyrokinetics
   $ # Use Github repo
   $ module load Git
   $ git clone https://github.com/pyro-kinetics/pyrokinetics ~/pyrokinetics
   $ cd ~/pyrokinetics
   $ pip install .

Some machines will already have gyrokinetics solvers installed and available via the
module system. If not, you'll also need to install these manually.

.. _sec-simple-hpc:

Simple Usage
------------

As a simple example, let's say we want to perform a bunch of similar simulations over
a range of proposed equilibrium geometries, such as those available
`here <https://doi.org/10.5281/zenodo.4643844>`_.

Rather than running Pyrokinetics directly on the login nodes, it is recommended to
collect all commands into an ``sbatch`` script, and to submit this to the Slurm job
scheduling system. Depending on your system, you may need to create this in your
'scratch' or 'work' space -- the area on your file system where it's permitted to save
large amounts of data, and where items may be periodically deleted to free up space.
We'll work in the directory ``~/scratch/pyrokinetics/``, and we'll begin by copying over
our equilibrium files and a GS2 template file:

.. code-block:: bash

   $ scp pyrokinetics_template.in *.geqdsk username@hpc.com:scratch/pyrokinetics/


The following script at ``~/scratch/pyrokinetics/pyrokinetics.sh`` will be used to
specify the job:

.. code-block:: bash

   #!/bin/bash

   # Slurm settings
   # --------------

   #SBATCH --job-name=pyrokinetics  # Job name
   #SBATCH --mail-user=me@email.com # Where to send mail reporting on the job
   #SBATCH --mail-type=END,FAIL     # Mail events (NONE, BEGIN, END, FAIL, ALL)
   #SBATCH --nodes=1                # Number of compute nodes to run on
   #SBATCH --ntasks-per-node=16     # Number of MPI processes to spawn per node
   #SBATCH --cpus-per-task=1        # Number of CPUs per MPI process
   #SBATCH --mem-per-cpu=8gb        # Memory allocated to each CPU
   #SBATCH --time=00:30:00          # Total time limit hrs:min:sec
   #SBATCH --output=%x_%j.log       # Log file for stdout outputs
   #SBATCH --error=%x_%j.err        # Log file for stderr outputs

   # The following SBATCH settings will vary depending on your machine:

   #SBATCH --partition=nodes
   #SBATCH --account=myaccountname

   # User settings
   # -------------

   # Set the location of the gs2 executable on your system
   gs2=$HOME/gs2/bin/gs2

   # Load the modules needed to run Pyrokinetics and GS2
   # These will vary depending on your system, and how GS2 was compiled
   module purge
   module load Python OpenMPI netCDF-Fortran FFTW.MPI/3.3.10-gompi-2023a

   # Activate the Python environment we installed Pyrokinetics to
   source $HOME/.local/venv/pyrokinetics/bin/activate

   # Generate inputs
   # ---------------

   pyro convert GS2 pyrokinetics_template.in --eq tdotp_lowq0.geqdsk --psi 0.5 \
     -o tdotp_lowq0.geqdsk.d/tdotp_lowq0.in
   pyro convert GS2 pyrokinetics_template.in --eq tdotp_highq0.geqdsk --psi 0.5 \
     -o tdotp_highq0.geqdsk.d/tdotp_highq0.in
   pyro convert GS2 pyrokinetics_template.in --eq tdotp_negtri.geqdsk --psi 0.5 \
     -o tdotp_negtri.geqdsk.d/tdotp_negtri.in

   # Perform runs
   # ------------

   srun $gs2 tdotp_lowq0.geqdsk.d/tdotp_lowq0.in
   srun $gs2 tdotp_highq0.geqdsk.d/tdotp_highq0.in
   srun $gs2 tdotp_negtri.geqdsk.d/tdotp_negtri.in

The script can be submitted to the job scheduler using:

.. code-block:: bash

   $ cd ~/scratch/pyrokinetics
   $ sbatch pyrokinetics.sh

You can check on its progress using:

.. code-block:: bash

   $ squeue --me

Assuming all goes well, this should generate three new GS2 input files -- one for each
equilibrium file -- and then run each of them sequentially with 16 MPI processes each.
The data will be available afterwards in the directories ``tdotp_*.geqdsk.d/``, and
should be copied back from the system for analysis:

.. code-block:: bash

   $ scp username@hpc.com:scratch/pyrokinetics/tdotp_lowq0.geqdsk.d/tdotp_lowq0.out.nc .

.. _sec-advanced-hpc:

Advanced job scheduling
-----------------------

Simple batch scripts should be sufficient for most jobs, but for some applications it
may be necessary to automate the process further. For example, we may have a very large
number of input files, or may not know how many runs we'll need to do in advance.
One option for these problems is to use Slurm job arrays and clever bash scripting.
In this example, we'll instead make use of the Python library
`QCG-PilotJob <https://qcg-pilotjob.readthedocs.io/en/develop/>`_, which allows you to
schedule jobs and manage resources from within a single Slurm allocation.

To begin, QCG-PilotJob should be installed to the Python environment we set up earlier:

.. code-block:: bash

   $ source ~/.local/venv/pyrokinetics/bin/activate
   $ pip install qcg-pilotjob

The job manager can be run using a Python script like the one shown below, which will be
saved to ``~/scratch/pyro_job/pyro_job.py``:

.. code-block:: python

   """Reads equilibrium files from the command line and schedules GS2 runs"""

   import argparse
   from pathlib import Path

   from pyrokinetics import Pyro
   from qcg.pilotjob.api.job import Jobs
   from qcg.pilotjob.api.manager import LocalManager


   def parse_args() -> argparse.Namespace:
       """Read command line arguments and return the result

       The command line application will take the following arguments:

       - Path to the GS2 executable
       - Path to the GS2 template file to use for all runs
       - List of paths to equilibrium files to process
       - psi_n, the flux surface coordinate to use in all simulations (optional)
       """
       parser = argparse.ArgumentParser(
           prog="pyro_job",
           description="Pyrokinetics job manager, runs within Slurm scheduling system",
       )
       parser.add_argument(
           "gs2_exe",
           type=Path,
           help="Path to pre-compiled GS2 executable on your system",
       )
       parser.add_argument(
           "gk_file",
           type=Path,
           help="Gyrokinetics input file used as basis for all runs",
       )
       parser.add_argument(
           "eq_files",
           type=Path,
           nargs="+",
           help="GEQDSK equilibrium files to simulate",
       )
       parser.add_argument(
           "--psi",
           type=float,
           default=0.5,
           help="Normalised psi at which to generate flux surfaces",
       )
       return parser.parse_args()


   def main() -> None:
       # Get command line arguments
       args = parse_args()

       # For each equilibrium file, get the path of a corresponding new GS2 input file
       # in its own directory
       eq_files = [path.resolve() for path in args.eq_files]
       new_dirs = [path.parent / f"{path.name}.d" for path in eq_files]
       gk_files = [d / f"{path.stem}.in" for d, path in zip(new_dirs, eq_files)]

       # Set up job queue
       jobs = Jobs()

       # Generate new input file for each equilibrium file, add to job queue
       gs2 = args.gs2_exe.resolve()
       template = args.gk_file.resolve()
       for gk_file, eq_file in zip(gk_files, eq_files):
           pyro = Pyro(gk_file=template, eq_file=eq_file)
           pyro.load_local_geometry(psi_n=args.psi)
           print("Generating input file:", gk_file)
           pyro.write_gk_file(gk_file, gk_code="GS2")
           jobs.add(
               name=str(gk_file.stem),         # Name each job
               exec="srun",                    # Run using srun...
               args=[str(gs2), str(gk_file)],  # ...with the GS2 exe and each input file
               stdout=str(gk_file.parent / f"{gk_file.stem}.log"),  # Log file name
               stderr=str(gk_file.parent / f"{gk_file.stem}.err"),  # Error file name
               numCores=8,                     # How many tasks per run
           )

       # Submit jobs and print stats
       manager = LocalManager()
       print("Available resources:", manager.resources())
       job_ids = manager.submit(jobs)
       print("Submitted jobs:", job_ids)
       job_status = manager.status(job_ids)
       print("Job status:", job_status)

       # Wait for jobs to complete and print final status
       manager.wait4(job_ids)
       job_info = manager.info(job_ids)
       print("Job detailed information:", job_info)
       manager.finish() # NB: This is needed!


   if __name__ == "__main__":
       main()

This can then be run with the batch script ``~/scratch/pyro_job/pyro_job.sh``:

.. code-block:: bash

   #!/bin/bash

   # Slurm settings
   # --------------

   #SBATCH --job-name=pyro_job      # Job name
   #SBATCH --mail-user=me@email.com # Where to send mail reporting on the job
   #SBATCH --mail-type=END,FAIL     # Mail events (NONE, BEGIN, END, FAIL, ALL)
   #SBATCH --nodes=1                # Number of compute nodes to run on
   #SBATCH --ntasks-per-node=24     # Number of MPI processes to spawn per node
   #SBATCH --cpus-per-task=1        # Number of CPUs per MPI process
   #SBATCH --mem-per-cpu=8gb        # Memory allocated to each CPU
   #SBATCH --time=00:30:00          # Total time limit hrs:min:sec
   #SBATCH --output=%x_%j.log       # Log file for stdout outputs
   #SBATCH --error=%x_%j.err        # Log file for stderr outputs
   #SBATCH --partition=nodes
   #SBATCH --account=myaccountname

   # User settings
   # -------------

   module purge
   module load Python OpenMPI netCDF-Fortran FFTW.MPI/3.3.10-gompi-2023a

   source $HOME/.local/venv/pyrokinetics/bin/activate

   # Perform runs
   # ------------

   gs2=$HOME/gs2/bin/gs2
   pyfile=$HOME/scratch/pyro_job/pyro_job.py
   template=$HOME/scratch/pyro_job/pyrokinetics_template.in

   python $pyfile $gs2 $template $HOME/scratch/pyro_job/*.geqdsk --psi 0.5

Note that we don't call ``srun`` explicitly here. In order to make these scripts more
suitable for automation, we haven't specified the names of the equilibrium files we wish
to process explicitly, and we've made use of full path names.

QCG-PilotJob is highly configurable, and can adapt to much more complex problems than
this. For example, rather than choosing exactly 8 cores for each job, we may instead set
a minimum and maximum core count for each job and let QCG-PilotJob manage the actual
allocations. It also contains features for restarting runs in case we time out.


.. _sec-dispatching-hpc:

Dispatching jobs remotely
-------------------------

To further assist with the automation of our HPC jobs, we can use a tool such as
`HPC Rocket <https://svenmarcus.github.io/hpc-rocket/>`_ to dispatch jobs to remote HPC
machines without the need to manually SSH in. This tool is well suited for use in
Continuous Integration (CI) pipelines, and can also be run from the command line.

HPC Rocket can be easily installed on the user's machine using:

.. code-block:: bash

   $ pip install hpc-rocket

Before using this, note that we'll still need to SSH into the HPC machine in order to
set up a Python environment and install all the software we'll need.

We can set up a remote job run by creating a new directory, adding the scripts in
:ref:`sec-advanced-hpc` and any equilibrium files we want, and then adding the file
``pyro_job.yaml``:

.. code-block:: yaml

   host: hpc.com
   user: username
   password: $PASSWORD

   copy:
     - from: pyrokinetics_template.in
       to: scratch/pyro_job/
       overwrite: true
     - from: ./*.geqdsk
       to: scratch/pyro_job/
       overwrite: true
     - from: pyro_job.sh
       to: scratch/pyro_job/
       overwrite: true
     - from: pyro_job.py
       to: scratch/pyro_job/
       overwrite: true

   collect:
     - from: scratch/pyro_job/*.d/*.out.nc
       to: .

   clean:
     - scratch/pyro_job/*.d/*.nc
     - scratch/pyro_job/*.geqdsk
     - scratch/pyro_job/pyrokinetics_template.in
     - scratch/pyro_job/pyro_job.sh
     - scratch/pyro_job/pyro_job.py

   sbatch: scratch/pyro_job/pyro_job.sh
   continue_if_job_fails: true

In order for this to work, we'll need to export our login password to the environment
variable ``PASSWORD`` before running. There are also options to automate the login
procedure using SSH keys.

With our YAML file set up, we can then dispatch a new remote job using:

.. code-block:: bash

   $ hpc-rocket launch --watch pyro_job.yaml
