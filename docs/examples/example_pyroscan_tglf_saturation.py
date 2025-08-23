#!/usr/bin/env python3
"""
Example: TGLF Saturation Rules with PyroScan

This example demonstrates how to apply TGLF saturation rules (SAT1, SAT2, SAT3)
to PyroScan objects from other gyrokinetic codes like CGYRO.

The TGLF saturation rules provide a way to calculate transport fluxes from 
linear gyrokinetic simulation data using well-validated saturation models.
"""

import numpy as np
import matplotlib.pyplot as plt
from pyrokinetics import Pyro, PyroScan, template_dir
from pyrokinetics.diagnostics.saturation_rules import SaturationRules

def create_example_pyroscan():
    """
    Create an example PyroScan object with CGYRO data for demonstration.
    In practice, this would come from actual CGYRO simulations.
    """
    # Load a CGYRO input template
    cgyro_file = template_dir / "outputs/CGYRO_linear/input.cgyro"
    
    # Create base Pyro object
    pyro = Pyro(gk_file=cgyro_file)
    
    # Create PyroScan for ky scan
    pyro_scan = PyroScan()
    pyro_scan.add_parameter_key("ky", "numerics", ["ky"])
    pyro_scan.set_base_pyro(pyro)
    
    # Define ky values to scan over
    ky_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5])
    
    # Set parameter values
    for ky in ky_values:
        pyro_scan.set_parameter_value("ky", ky)
    
    # In a real application, you would run the scans here
    # For this example, we'll create mock output data
    
    print(f"Created PyroScan with {len(ky_values)} ky points: {ky_values}")
    return pyro_scan, ky_values

def create_mock_gk_output(ky_values):
    """
    Create mock gk_output data that resembles typical gyrokinetic results.
    In practice, this data would come from actual CGYRO simulations.
    """
    import xarray as xr
    
    nky = len(ky_values)
    
    # Create coordinates
    ky_coords = xr.DataArray(ky_values, dims=["ky"], name="ky")
    species_coords = xr.DataArray(["ion1", "electron"], dims=["species"], name="species")
    field_coords = xr.DataArray(["phi"], dims=["field"], name="field")
    
    # Create realistic growth rates (ITG-like scaling)
    # Peak around ky ~ 0.3, then decrease
    ky_peak = 0.3
    gamma_max = 0.15
    growth_rates = gamma_max * np.exp(-((ky_values - ky_peak) / 0.4)**2)
    growth_rates = np.maximum(growth_rates, 0.01)  # Minimum growth rate
    
    growth_rate = xr.DataArray(
        growth_rates,
        dims=["ky"],
        coords={"ky": ky_coords}
    )
    
    # Create mock quasi-linear flux data
    # Shape: [field, species, ky]
    
    # Particle flux: moderate values, peaked around ky_peak
    particle_base = 0.1 * np.exp(-((ky_values - ky_peak) / 0.5)**2)
    particle_flux_data = np.zeros((1, 2, nky))
    particle_flux_data[0, 0, :] = particle_base * 1.2  # Ion flux
    particle_flux_data[0, 1, :] = particle_base * 0.8  # Electron flux
    
    # Heat flux: larger values, similar ky dependence
    heat_base = 0.5 * np.exp(-((ky_values - ky_peak) / 0.5)**2)
    heat_flux_data = np.zeros((1, 2, nky))
    heat_flux_data[0, 0, :] = heat_base * 1.5  # Ion heat flux
    heat_flux_data[0, 1, :] = heat_base * 1.0  # Electron heat flux
    
    # Momentum flux: smaller values
    momentum_flux_data = np.zeros((1, 2, nky))
    momentum_flux_data[0, 0, :] = particle_base * 0.1  # Ion momentum
    momentum_flux_data[0, 1, :] = particle_base * 0.05  # Electron momentum
    
    # Create xarray DataArrays
    particle_flux = xr.DataArray(
        particle_flux_data,
        dims=["field", "species", "ky"],
        coords={"field": field_coords, "species": species_coords, "ky": ky_coords}
    )
    
    heat_flux = xr.DataArray(
        heat_flux_data,
        dims=["field", "species", "ky"],
        coords={"field": field_coords, "species": species_coords, "ky": ky_coords}
    )
    
    momentum_flux = xr.DataArray(
        momentum_flux_data,
        dims=["field", "species", "ky"],
        coords={"field": field_coords, "species": species_coords, "ky": ky_coords}
    )
    
    # Create dataset
    gk_output = xr.Dataset({
        "growth_rate": growth_rate,
        "particle": particle_flux,
        "heat": heat_flux,
        "momentum": momentum_flux
    })
    
    print("Created mock gk_output with realistic ITG-like scaling")
    return gk_output

def apply_tglf_saturation(pyro_scan, sat_rules=[1, 2, 3]):
    """
    Apply different TGLF saturation rules to the PyroScan data.
    """
    # Create SaturationRules object
    saturation = SaturationRules(pyro_scan)
    
    results = {}
    
    for sat_rule in sat_rules:
        print(f"\nApplying TGLF SAT{sat_rule} saturation rule...")
        
        result = saturation.tglf_saturation(
            sat_rule=sat_rule,
            output_convention="pyrokinetics",
            units="GYRO",
            alpha_zf=1.0,
            vexb_shear=0.0,  # No ExB shear for this example
            alpha_e=1.0,
            rlnp_cutoff=18.0
        )
        
        results[f"SAT{sat_rule}"] = result
        
        # Print summary
        print(f"  Ion heat flux: {result['heat'].sel(species='ion1').values:.4f}")
        print(f"  Electron heat flux: {result['heat'].sel(species='electron').values:.4f}")
        print(f"  Total heat flux: {result['heat'].sum().values:.4f}")
    
    return results

def plot_results(ky_values, original_data, sat_results):
    """
    Create comparison plots of original quasi-linear data vs. saturated fluxes.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Growth rates
    ax1.plot(ky_values, original_data["growth_rate"].values, 'k-', linewidth=2, label="Growth rate")
    ax1.set_xlabel("ky")
    ax1.set_ylabel("γ")
    ax1.set_title("Growth Rate Spectrum")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Original quasi-linear heat fluxes
    ion_ql = original_data["heat"].sel(field="phi", species="ion1").values
    electron_ql = original_data["heat"].sel(field="phi", species="electron").values
    
    ax2.plot(ky_values, ion_ql, 'r-', label="Ion (QL)")
    ax2.plot(ky_values, electron_ql, 'b-', label="Electron (QL)")
    ax2.set_xlabel("ky")
    ax2.set_ylabel("Heat Flux (QL)")
    ax2.set_title("Original Quasi-Linear Heat Flux")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Saturated heat fluxes comparison
    colors = ['red', 'green', 'blue']
    for i, (rule_name, result) in enumerate(sat_results.items()):
        ion_sat = result["heat"].sel(species="ion1").values
        electron_sat = result["heat"].sel(species="electron").values
        total_sat = ion_sat + electron_sat
        
        ax3.bar(i*0.8 + 0.1, ion_sat, 0.3, color=colors[i], alpha=0.7, label=f"{rule_name} Ion")
        ax3.bar(i*0.8 + 0.4, electron_sat, 0.3, color=colors[i], alpha=0.4, label=f"{rule_name} Electron")
    
    ax3.set_xlabel("Saturation Rule")
    ax3.set_ylabel("Saturated Heat Flux")
    ax3.set_title("TGLF Saturated Heat Fluxes")
    ax3.set_xticks([0.25, 1.05, 1.85])
    ax3.set_xticklabels(["SAT1", "SAT2", "SAT3"])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Flux comparison table
    ax4.axis('off')
    
    # Create comparison table
    table_data = []
    headers = ["Rule", "Ion Heat", "Electron Heat", "Total Heat", "Ion Particle"]
    
    for rule_name, result in sat_results.items():
        row = [
            rule_name,
            f"{result['heat'].sel(species='ion1').values:.4f}",
            f"{result['heat'].sel(species='electron').values:.4f}",
            f"{result['heat'].sum().values:.4f}",
            f"{result['particle'].sel(species='ion1').values:.4f}"
        ]
        table_data.append(row)
    
    table = ax4.table(cellText=table_data, colLabels=headers, 
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax4.set_title("Flux Comparison Table", pad=20)
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main example function demonstrating TGLF saturation with PyroScan.
    """
    print("TGLF Saturation Rules with PyroScan Example")
    print("=" * 50)
    
    try:
        # Step 1: Create example PyroScan
        print("Step 1: Creating example PyroScan...")
        pyro_scan, ky_values = create_example_pyroscan()
        
        # Step 2: Create mock gk_output data
        print("\nStep 2: Creating mock gyrokinetic output data...")
        mock_output = create_mock_gk_output(ky_values)
        
        # Attach mock data to pyro_scan
        pyro_scan.gk_output = mock_output
        
        # Step 3: Apply TGLF saturation rules
        print("\nStep 3: Applying TGLF saturation rules...")
        sat_results = apply_tglf_saturation(pyro_scan, sat_rules=[1, 2, 3])
        
        # Step 4: Display results
        print("\nStep 4: Results summary:")
        print("-" * 30)
        for rule_name, result in sat_results.items():
            total_heat = result["heat"].sum().values
            total_particle = result["particle"].sum().values
            print(f"{rule_name}: Total heat flux = {total_heat:.4f}, "
                  f"Total particle flux = {total_particle:.4f}")
        
        # Step 5: Create plots
        print("\nStep 5: Creating comparison plots...")
        plot_results(ky_values, mock_output, sat_results)
        
        print("\nExample completed successfully!")
        print("\nKey takeaways:")
        print("- Different TGLF saturation rules give different transport predictions")
        print("- SAT2 and SAT3 are generally more conservative than SAT1")
        print("- The method can be applied to any PyroScan with ky-spectrum data")
        print("- Real applications would use actual CGYRO/GS2/GENE simulation data")
        
    except Exception as e:
        print(f"Example failed: {e}")
        print("This may be due to missing template files or dependencies.")
        print("In a real application, you would have your own simulation data.")

if __name__ == "__main__":
    main()
