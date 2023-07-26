from pyrokinetics import Pyro, template_dir
import os
import pathlib
from typing import Union


def main(base_path: Union[os.PathLike, str] = "."):
    
    # Equilibrium file
    eq_file = template_dir / "test.geqdsk"
    
    # Kinetics data file
    kinetics_file = template_dir / "pfile.txt"
    
    # Load up pyro object
    pyro = Pyro(
        eq_file=eq_file,
        eq_type="GEQDSK",
        kinetics_file=kinetics_file,
        kinetics_type="pFile",
    )
    
    pyro.local_geometry = "Miller"
    
    # Generate local Miller parameters at psi_n=0.5
    pyro.load_local_geometry(psi_n=0.5)
    pyro.load_local_species(psi_n=0.5)
    
    pyro.gk_code = "GS2"

    base_path = pathlib.Path(base_path)

    # Write out GK
    pyro.write_gk_file(file_name=base_path / "test_pfile.gs2")
    
    pyro.write_gk_file(file_name=base_path / "test_pfile.gene", gk_code="GENE")

    pyro.write_gk_file(file_name=base_path / "test_pfile.tglf", gk_code="TGLF")
    
    pyro.write_gk_file(file_name=base_path / "test_pfile.gene", gk_code="GENE")

    pyro.write_gk_file(file_name=base_path / "test_pfile.cgyro", gk_code="CGYRO")

    return pyro
    
if __name__ == "__main__":
    main()
