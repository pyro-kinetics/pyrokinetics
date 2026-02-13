from pyrokinetics import pyro, template_dir

# Saving pyro gk outputs to netcdf files and loading from then can save time
# especially if loading large files

path = template_dir / "outputs" / "CGYRO_linear"
pyro = Pyro(gk_file=path / "input.cgyro")
pyro.load_gk_output()
pyro.gk_output.to_netcdf(path / "linear_cgyro.nc")

new_pyro = Pyro(gk_file=path / "input.cgyro")
new_pyro.load_gk_output(netcdf_file=path / "linear_cgyro.nc")
