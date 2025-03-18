from pyrokinetics import Pyro
import numpy as np
import matplotlib.pyplot as plt
import os
import sys


# Function to handle legend clicks
def on_legend_click(event):
    legline = event.artist
    origline = None

    for leg, line in zip(legend.get_lines(), [line1, line2, line3]):
        if leg == legline:
            origline = line
            break

    if origline is not None:
        visible = not origline.get_visible()
        origline.set_visible(visible)
        legline.set_alpha(1.0 if visible else 0.2) # change legend line opacity
        fig.canvas.draw()


def load_pyro_from_dirs(base_dir):
    """
    Loads Pyro objects from subdirectories within the base_dir.

    Args:
        base_dir: The base directory containing subdirectories with input files.

    Returns:
        A dictionary where keys are directory names (or numbered names if needed)
        and values are Pyro objects.
    """
    pyro_objects = {}
    for subdir in sorted(os.listdir(base_dir)):  # Sort for consistent order
        subdir_path = os.path.join(base_dir, subdir)
        if os.path.isdir(subdir_path):
            input_file = os.path.join(subdir_path, "input.tglf")  # Assuming "input.tglf" is the file name
            if os.path.exists(input_file):
                try:
                    pyro_objects[subdir] = Pyro(gk_file=input_file, gk_code="TGLF")
                except Exception as e:
                    print(f"Error loading Pyro from {subdir}: {e}")
            else:
                print(f"Warning: input.tglf not found in {subdir}")
    return pyro_objects

def load_numbered_pyro_from_dirs(base_dir, prefix="dir"):
    """
    Loads Pyro objects from subdirectories with numbered names.

    Args:
        base_dir: The base directory containing numbered subdirectories.
        prefix: prefix of each subdirectory, example: "dir"

    Returns:
        A dictionary where keys are numbered names (e.g., "dir1", "dir2")
        and values are Pyro objects.
    """
    pyro_objects = {}
    i = 1
    while True:
        subdir_name = f"{prefix}{i}"
        subdir_path = os.path.join(base_dir, subdir_name)
        if not os.path.isdir(subdir_path):
            break  # Stop when a numbered directory doesn't exist

        input_file = os.path.join(subdir_path, "input.tglf")
        if os.path.exists(input_file):
            try:
                pyro_objects[subdir_name] = Pyro(gk_file=input_file, gk_code="TGLF")
            except Exception as e:
                print(f"Error loading Pyro from {subdir_name}: {e}")
        else:
            print(f"Warning: input.tglf not found in {subdir_name}")

        i += 1
    return pyro_objects

# Example usage:
base_directory = "./"  # Replace with your base directory

# Load from subdirectories with arbitrary names
pyro_list = load_pyro_from_dirs(base_directory)
for dir_name, pyro_obj in pyro_list.items():
    pyro_obj.load_gk_output()
    
    ax1=plt.figure(1)
    growth_rate = pyro_obj.gk_output.data["growth_rate"]
    growth_rate[:,0].plot(label=pyro_obj.gk_file)
    
    ax2=plt.figure(2)    
    mode_freq = pyro_obj.gk_output.data["mode_frequency"]
    mode_freq[:,0].plot(label=pyro_obj.gk_file)
    
    #growth_rate[:,1].plot(label='1')  
    print(f"Loaded Pyro from {dir_name}: {pyro_obj.gk_file}")


# Create the legend
#legend = ax.legend()

# Connect the click event to the legend
#legend.get_frame().set_picker(True)  # Enable picking on the legend frame
#fig.canvas.mpl_connect('pick_event', on_legend_click)


plt.legend()
plt.show()

# Load from numbered subdirectories (e.g., dir1, dir2, ...)
#numbered_pyro_instances = load_numbered_pyro_from_dirs(base_directory, prefix="dir")
#for dir_name, pyro_obj in numbered_pyro_instances.items():
#    print(f"Loaded Pyro from {dir_name}: {pyro_obj.gk_file}")



# Plot dominant eigenfunction
#try:
#  dominant = pyro.gk_output.data["eigenfunctions"].isel(mode=0)#

#  np.real(dominant.sel(field="phi")).plot(x="theta", marker="x")
#  np.imag(dominant.sel(field="phi")).plot(x="theta", marker="o")
#  plt.show()

#  np.real(dominant.sel(field="apar")).plot(marker="x")
#  np.imag(dominant.sel(field="apar")).plot(marker="o")
#  plt.show()
#except:
#  print('no eigenfunctions present, rerun TGLF with WRITE_WAVEFUNCTION_FLAG=T and USE_TRANSPORT_MODEL=F ??')

# Plot subdominant eigenfunction
#try:
#  subdominant = pyro.gk_output.data["eigenfunctions"].isel(mode=1)
#  np.real(subdominant.sel(field="phi")).plot(marker="x")
#  np.imag(subdominant.sel(field="phi")).plot(marker="o")
#  plt.show()

#  np.real(subdominant.sel(field="apar")).plot(marker="x")
#  np.imag(subdominant.sel(field="apar")).plot(marker="o")
#  plt.show()
#except:
#  print('no eignenfunctions present, rerun TGLF with WRITE_WAVEFUNCTION_FLAG=T')

