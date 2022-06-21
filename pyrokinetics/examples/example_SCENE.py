from pyrokinetics import Pyro, template_dir
import os
import pathlib
from typing import Union


def main(base_path: Union[os.PathLike, str] = "."):
    # Equilibrium file
    eq_file = template_dir / "test.geqdsk"

    # Kinetics data file
    kinetics_file = template_dir / "scene.cdf"

    # CGYRO template
    gk_file = template_dir / "input.cgyro"

    pyro = Pyro(
        gk_file=gk_file,
        eq_file=eq_file,
        kinetics_file=kinetics_file,
    )

    # Generate local Miller parameters at psi_n=0.5
    pyro.load_local(psi_n=0.5, local_geometry="Miller")

    base_path = pathlib.Path(base_path)

    # Write CGYRO input file using default template
    pyro.write_gk_file(file_name=base_path / "test_scene.cgyro")

    # Write single GS2 input file, specifying the code type
    # in the call.
    pyro.write_gk_file(file_name=base_path / "test_scene.gs2", gk_code="GS2")

    # Write single GENE input file, specifying the code type
    # in the call.
    pyro.write_gk_file(file_name=base_path / "test_scene.gene", gk_code="GENE")

    return pyro


if __name__ == "__main__":
    main()
