import sys
import numpy as np
from pydmd import DMD
from pyrokinetics import Pyro

file_name = sys.argv[1]

pyro = Pyro(gk_file=file_name)

pyro.load_gk_output()
data = pyro.gk_output
growth_rate = data["growth_rate"]
mode_frequency = data["mode_frequency"]
eigenvalues = data["eigenvalues"].sel(kx=0).isel(time=-1, ky=0)
theta = pyro.gk_output.data["theta"].data
time = pyro.gk_output.data["time"].data
ntime_sim = len(time)

nmodes = 2
ntimes = 129
delt = time[1] - time[0]

eigenvalues_dict = {}

for field in data["field"].data:
    eigenfunction = (
        pyro.gk_output.data[field]
        .isel(kx=0, ky=0)
        .isel(time=slice(ntime_sim - ntimes, ntime_sim - 1))
    )
    eigenfunction = eigenfunction.data.m
    dmd = DMD(svd_rank=nmodes, exact=True)
    dmd.fit(eigenfunction)

    eigenvalues_result = np.log(dmd.eigs) * 1j / delt
    eigenvalues_dict[str(field)] = eigenvalues_result


def group_and_average_common_tolerant(data, relative_tolerance=0.1):
    # Step 1: Flatten all numbers into a single unique set
    all_nums = sorted(
        {num for values in data.values() for num in values},
        key=lambda z: (z.real, z.imag),
    )

    # Step 2: Group numbers within relative tolerance
    grouped = []
    while all_nums:
        group = [all_nums.pop(0)]
        i = 0
        while i < len(all_nums):
            ref = group[0]  # Reference for relative comparison
            if abs(all_nums[i] - ref) / (abs(ref) + 1e-10) <= relative_tolerance:
                group.append(all_nums.pop(i))  # Add to group
            else:
                i += 1
        # Store the average of this group
        grouped.append(sum(group) / len(group))

    # Step 3: Keep only the groups that appear (within tolerance) in every dictionary item
    def is_group_in_all_items(avg):
        """Returns True if every dictionary item has at least one number close to avg."""
        return all(
            any(
                abs(num - avg) / (abs(avg) + 1e-10) <= relative_tolerance
                for num in values
            )
            for values in data.values()
        )

    final_averages = np.array([avg for avg in grouped if is_group_in_all_items(avg)])

    sorted_order = np.argsort(final_averages.imag)

    eigenvalues_result = final_averages[sorted_order][::-1]
    return eigenvalues_result


new_eigenvalues = group_and_average_common_tolerant(
    eigenvalues_dict, relative_tolerance=0.01
)

match_dominant_freq = np.isclose(
    eigenvalues.data.m.real, new_eigenvalues[0].real, rtol=0.05
)
match_dominant_growth = np.isclose(
    eigenvalues.data.m.imag, new_eigenvalues[0].imag, rtol=0.05
)

if match_dominant_freq:
    print("DMD found a matching dominant mode frequency")
else:
    print("DMD did not find a matching dominant mode frequency")

if match_dominant_growth:
    print("DMD found a matching dominant growth rate")
else:
    print("DMD did not find a matching dominant growth rate")


print(f"Pyro dominant eigenvalue: {eigenvalues.data.m}")
nmodes_found = len(new_eigenvalues)
print(f"Pyro DMD found {nmodes_found} unstable modes")

for i, ev in enumerate(new_eigenvalues):
    print(f"Mode {i}: {ev}")
