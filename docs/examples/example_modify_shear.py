from pyrokinetics import Pyro, template_dir
from pyrokinetics.diagnostics.neoclassical import Redl2021

# Solves <B^2> f' / μ0 + fp' = <Jtot.B> = <Jbs.B> + <Jext.B>
# Scales beta/gradients keeping boostrap current consistent and
# modifies external current to scale like T/n

run = "TRANSP"

if run == "TRANSP":
    # Path to files
    eq_file = template_dir / "transp.cdf"
    kinetics_file = template_dir / "transp.cdf"

    # Load up pyro object
    pyro = Pyro(
        eq_file=eq_file,
        eq_type="TRANSP",
        eq_kwargs={"time_index": -1},
        kinetics_file=kinetics_file,
        kinetics_type="TRANSP",
        kinetics_kwargs={"time_index": -1},
    )

elif run == "SCENE":
    # Path to files
    eq_file = template_dir / "test.geqdsk"
    kinetics_file = template_dir / "scene.cdf"

    # Load up pyro object
    pyro = Pyro(
        eq_file=eq_file,
        eq_type="GEQDSK",
        kinetics_file=kinetics_file,
        kinetics_type="SCENE",
    )

elif run == "JETTO":
    # Path to files
    eq_file = template_dir / "transp_eq.geqdsk"
    kinetics_file = template_dir / "jetto.jsp"

    # Load up pyro object
    pyro = Pyro(
        eq_file=eq_file,
        eq_type="GEQDSK",
        kinetics_file=kinetics_file,
        kinetics_type="JETTO",
        kinetics_kwargs={"time_index": -1},
    )

# Load local parameters
pyro.load_local(psi_n=0.5, local_geometry="MXH", show_fit=False)

# Need Numerics object for beta
pyro.load_numerics(load_geometry_species_data=True, ntheta=512, apar=True, bpar=True)

# Ensure beta' matches
pyro.enforce_consistent_beta_prime()

# Optional! Load from file (ensure full consistency in metric.py)
pyro.write_gk_file("test.gene", gk_code="GENE")
pyro = Pyro(gk_file="test.gene")
lg = pyro.local_geometry

old_shat = lg.shat
old_Fprime = lg.get_f_prime()

# Get original bootstrap/external current
old_redl = Redl2021(pyro, ntheta=512)
old_jbsdotb = old_redl.JbsdotB
old_jextdotb = old_redl.JextdotB
old_jdotb = old_redl.JdotB

# Modify equilibrium/gradients
n_factor = 1.2
T_factor = 1.0

beta_scale = n_factor * T_factor
pyro.numerics.beta = beta_scale * pyro.norms.beta

alt_scale = 1.2
for species in pyro.local_species.names:
    pyro.local_species[species].inverse_lt *= alt_scale

pyro.enforce_consistent_beta_prime()

# Get new bootstrap/external curent
new_redl = Redl2021(pyro, ntheta=512)
new_jbsdotb = new_redl.JbsdotB
new_jextdotb = new_redl.JextdotB
new_jdotb = new_redl.JdotB

# Set new total current as old external + new bootstrap
mod_jbsdotb = new_jbsdotb
mod_jextdotb = old_jextdotb * T_factor / n_factor
mod_jdotb = mod_jextdotb + mod_jbsdotb

# Determine new F' given new total current
new_Fprime = new_redl.get_Fprime_from_total_current(mod_jdotb)
old_shat = pyro.local_geometry.shat

# Determine new shat given new F'
new_shat = lg.get_s_hat(new_Fprime)
lg.shat = new_shat

# Calculate new JdotB to ensure it matches expectation
update_redl = Redl2021(pyro, ntheta=512)
update_jbsdotb = update_redl.JbsdotB
update_jextdotb = update_redl.JextdotB
update_jdotb = update_redl.JdotB

print(f"Increase beta by {((beta_scale - 1)*100):.1f}%")
print(f"Increase a/LT by {((alt_scale - 1)*100):.1f}%")


print("Jbs before", old_jbsdotb)
print("Jbs after ", new_jbsdotb)
print()
print("Jext before", old_jextdotb)
print("Jext mod   ", mod_jextdotb)
print()
print("Jtot before", old_jdotb)
print("Jtot after ", new_jdotb)
print("Jtot mod   ", mod_jdotb)
print("Jtot update", update_jdotb)
print()
print("Original F'", old_Fprime)
print("New F'     ", new_Fprime)
print()
print("Original s hat", old_shat)
print("New s hat     ", new_shat)
