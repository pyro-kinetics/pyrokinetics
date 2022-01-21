import pyrokinetics

# Set up Pyro run
gs2_template = pyrokinetics.template_dir / "input.gs2"
pyro = pyrokinetics.Pyro(gk_file=gs2_template, gk_type="GS2")
pyro.write_gk_file(file_name="step.gs2")

# Set up cluster info
pyro.set_cluster(
    cluster_name="Marconi",
    scheduler="SLURM",
    job_name="my_test_job",
    wall_time=24.0,
    account="TEST_ACCOUNT",
    partition="TEST_PARTITION",
    output="batch%j.out",
    error="batch%j.err",
    n_mpi=48,
    n_nodes=1,
    n_omp=1,
    email_type="END",
    email="bhavin.s.patel@ukaea.uk",
)

# Write submission script
pyro.cluster.write_submission_script(pyro)

# Submit to cluster
pyro.cluster.submit_job(pyro)
