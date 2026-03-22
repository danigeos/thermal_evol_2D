#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def generate_thermal_plot(D=300):
    """
    Generates a 2D thermal profile of the Martian subsurface following a meteorite impact.
    
    Args:
        D (int): Diameter of the meteorite in meters.
    """
    # --- PHYSICAL PARAMETERS ---
    # Melt Radius (Rm) is approximately 1.7 * D for typical asteroid velocities
    Rm = 1.7 * D  
    # Attenuation coefficient (n) - Standard for Martian Basalt (1.5 to 2.0)
    # Higher n means energy is absorbed faster (shorter anomaly)
    n = 1.6   
    # Reference Temperature Increase at the edge of the melt zone (Celsius)
    Tm = 1200 

    # --- MARS GEOTHERM PARAMETERS ---
    # Surface temperature typical for Mars (Celsius)
    T_surf = -58  
    # Geothermal gradient (0.01 C/m = 10 C/km)
    grad = 0.01   

    # --- GRID SETUP ---
    # We visualize the upper 5km of the crust across a 10km horizontal span
    resolution = 500
    x_range = np.linspace(-3000, 8000, resolution)
    z_range = np.linspace(0, 6000, resolution)
    X, Z = np.meshgrid(x_range, z_range)
    
    # Calculate radial distance R from the impact origin (0,0)
    R = np.sqrt(X**2 + Z**2)

    # --- THERMAL CALCULATIONS ---
    # 1. Shock Temperature Increase (Delta T)
    # Power-law decay based on Gault-Heitowit shock models
    R_safe = np.where(R < Rm, Rm, R)
    delta_T = Tm * (Rm / R_safe)**(2 * n)
    
    # Clip to maximum melt temperature at the impact core
    delta_T = np.clip(delta_T, 0, Tm)

    # 2. Background Geotherm (Temperature before impact)
    T_geo = T_surf + grad * Z

    # 3. Total Temperature (Geotherm + Shock Heat)
    T_total = T_geo + delta_T

    # --- VISUALIZATION ---
    plt.figure(figsize=(12, 10)) 

    # High-granularity levels for the -60 to +100 range to see subtle anomalies
    levels = np.concatenate([
        np.linspace(-60, 0, 13),      # -60 to 0 (5 degree steps)
        np.linspace(5, 100, 20),      # 5 to 100 (5 degree steps)
        [200, 400, 600, 800, 1000, 1200]
    ])

    # nipy_spectral is excellent for distinguishing cold/warm transitions on Mars
    contour = plt.contourf(X/1000, Z/1000, T_total, levels=levels, cmap='nipy_spectral', extend='both')
    
    # Colorbar configuration
    cbar = plt.colorbar(contour, fraction=0.046, pad=0.04)
    cbar.set_label('Total Temperature (°C)', fontsize=12)

    # Contour lines every 10C between -60 and +10
    # We plot the 0C line separately to make it bold
    levels_gen = [t for t in range(-60, 11, 10) if t != 0]
    lines_gen = plt.contour(X/1000, Z/1000, T_total, levels=levels_gen, colors='white', linewidths=0.8, alpha=0.7)
    plt.clabel(lines_gen, inline=True, fontsize=9, fmt='%d')

    # 0°C Isotherm (Bold)
    line_0 = plt.contour(X/1000, Z/1000, T_total, levels=[0], colors='white', linewidths=2.5)
    plt.clabel(line_0, inline=True, fontsize=12, fmt='0°C')

    # Formatting: Equal aspect ratio ensures a 1km horizontal distance looks same as 1km depth
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().invert_yaxis()
    
    plt.title(f'Martian Subsurface Thermal Profile | Meteorite D = {D}m\n'
              f'Geotherm: {T_surf}°C + {grad*1000}°C/km | Equal 1:1 Aspect Ratio', fontsize=14)
    plt.xlabel('Horizontal Distance from Impact (km)', fontsize=12)
    plt.ylabel('Depth (km)', fontsize=12)
    plt.grid(alpha=0.3, linestyle=':')

    plt.tight_layout()
    output_file = os.path.splitext(os.path.basename(__file__))[0] + ".png"
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")

    # Export temperature field to a text file
    txt_file = os.path.splitext(os.path.basename(__file__))[0] + ".txt"
    np.savetxt(txt_file, np.column_stack((X.flatten(), Z.flatten(), T_total.flatten())), fmt='%.6e', header=f'Meteorite Diameter: {D}m\nx(m) z(m) T(C)')
    print(f"Field data saved to {txt_file}")

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a 2D thermal profile of the Martian subsurface following a meteorite impact.")
    parser.add_argument("-D", "--diameter", type=float, default=300.0, 
                        help="Diameter of the meteorite in meters (default: 300). Use ~800m to see 10C anomaly at 5km depth.")
    args = parser.parse_args()
    
    generate_thermal_plot(args.diameter)