from pyrokinetics import Pyro, template_dir


def main():

    # Equilibrium file
    eq_file = template_dir / "test.geqdsk"

    # Kinetics data file
    kinetics_file = template_dir / "jetto.jsp"

    # Load up pyro object
    pyro = Pyro(
        eq_file=eq_file,
        kinetics_file=kinetics_file,
        kinetics_type="JETTO",
        kinetics_kwargs={"time": 550},
    )

    # Generate local parameters at psi_n=0.5
    pyro.load_local(psi_n=0.5, local_geometry="Miller")

    # merge species 'impurity1' into 'deuterium'
    pyro.local_species.merge_species(
        "deuterium",
        ["impurity1"],
        keep_base_species_mass=False,
        keep_base_species_z=False,
    )

    # write to file
    pyro.write_gk_file(file_name="input.gene", gk_code="GENE")

    return pyro


if __name__ == "__main__":

    main()
