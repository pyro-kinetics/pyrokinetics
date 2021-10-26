import os
from cleverdict import CleverDict


class Cluster(CleverDict):
    r"""
    Cluster object describing cluster where Pyro objects can be run

    Data stored in a CleverDict object

    Attributes
    ----------
    cluster_name : String
        Name of the cluster
    scheduler : String
        Scheduler type i.e SLURM, PBS etc.
    job_name : String
        Name of job to be submitted to cluster
    wall_time : Float
        Wall time in hours for job
    n_mpi : Integer
        Total no. of MPI processes to be used
    n_omp : Integer
        Total no. of OMP processes to be used
    n_nodes : Integer
        Total no. of nodes to be used
    email_type : String
        Email type describing when to send emails
    email : String
        Email of user
    memory_req : Float
        Memory requirements of job in MB
    output : String
        File name for standard output for the job
    error : String
        File name for error output for the job
    run_command : String
        Run command for the gk_code
    account : String
        Account name on the cluster
    partition : String
        Partition name on the cluster
    """

    def __init__(self, **kwargs):

        self.default()

        for kw in kwargs:
            self[kw] = kwargs[kw]

        self.supported_schedulers = [
            "SLURM",
        ]

    def default(self):

        cluster = {
            "cluster_name": None,
            "scheduler": "SLURM",
            "job_name": "Pyro_run",
            "wall_time": 24.0,
            "n_mpi": 8,
            "n_omp": 1,
            "n_nodes": 1,
            "email_type": "ALL",
            "email": None,
            "memory_req": None,
            "output": "batch%j.out",
            "error": "batch%j.err",
            "run_command": None,
            "account": None,
            "partition": None,
        }

        super(Cluster, self).__init__(cluster)

    def use_marconi(
        self, job_name, n_mpi, n_omp, n_nodes, email_type, email, account, partition
    ):
        r"""

        Set up Cluster object to Marconi

        Parameters
        ----------
        cluster_name
        job_name
        n_mpi
        n_omp
        n_nodes
        email_type
        email
        account
        partition

        """

        self.cluster_name = "MARCONI"
        self.scheduler = "SLURM"

        self.job_name = job_name
        self.n_mpi = n_mpi
        self.n_omp = n_omp
        self.n_nodes = n_nodes
        self.email_type = email_type
        self.email = email
        self.account = account
        self.partition = partition

    def write_submission_script(self, pyro):
        r"""

        Writes submission script for cluster

        Parameters
        ----------
        pyro

        """
        print(self.scheduler)
        if self.scheduler not in self.supported_schedulers:
            raise NotImplementedError(
                f"Writing submission scripts for a {self.scheduler} not yet supported"
            )
        elif self.scheduler == "SLURM":
            self.write_submission_script_slurm(pyro)

    def write_submission_script_slurm(self, pyro):
        r"""

        Writes submission scripts for SLURM scheduler
        Parameters
        ----------
        pyro

        """

        job_file = os.path.join(pyro.run_directory, "batch.src")
        self.job_file = job_file

        self.run_command = pyro.gk_code.run_command(pyro)

        with open(job_file, "w+") as f:
            f.writelines("#!/bin/bash\n")
            f.writelines(f"#SBATCH --job-name={self.job_name}.job\n")
            f.writelines(f"#SBATCH --account={self.account}\n")
            f.writelines(f"#SBATCH --partition={self.partition}\n")
            f.writelines(f"#SBATCH --output={self.output}\n")
            f.writelines(f"#SBATCH --error={self.error}\n")

            hours = int(self.wall_time)
            minutes = int((self.wall_time % 1) * 60)
            f.writelines(f"#SBATCH --time={hours}:{minutes:02d}:00\n")

            if self.memory_req is not None:
                f.writelines(f"#SBATCH --mem={self.memory_req}\n")

            f.writelines(f"#SBATCH --ntasks={self.n_mpi}\n")
            f.writelines(f"#SBATCH --cpus-per-task={self.n_omp}\n")
            f.writelines(f"#SBATCH --nodes={self.n_nodes}\n")
            f.writelines(f"#SBATCH --mail-type={self.email_type}\n")
            f.writelines(f"#SBATCH --mail-user={self.email}\n")

            f.writelines(f"{self.run_command}")

        f.close()

    def submit_job(self, pyro):
        r"""
        Submits job to cluster

        Parameters
        ----------
        pyro

        """

        if not os.path.exists(self.job_file):
            self.write_submission_script(pyro)

        with cd(pyro.run_directory):
            os.system("sbatch batch.src")


class cd:
    r"""
    Context manager for changing the current working directory
    """

    def __init__(self, new_path):
        self.newPath = os.path.expanduser(new_path)

    def __enter__(self):
        self.saved_path = os.getcwd()
        os.chdir(self.new_path)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.saved_path)
