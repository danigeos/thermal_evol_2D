#!/usr/bin/env python3

"""
Dike Isotherm Depth Plotter (Updated for thermal_evol.results.txt)
Visualizes two datasets using only contours:
1. Solid lines: Minimum depth of 0°C isotherm (column 5)
2. Dashed lines: Minimum depth of organisms (column 7)

Includes updated geological annotations for Cerberus F. and Elysium F.
as semitransparent areas with +/- 20% width.
Labels are placed inside at the top-center of the areas.
Special "Surface" zone highlighted for depths < 25m.
"""

import os
import sys
import argparse

# Check for required libraries
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.interpolate import griddata
except ImportError:
    print("Error: Missing required libraries.")
    print("Please run: pip3 install pandas numpy matplotlib scipy")
    sys.exit(1)

def load_and_clean_data(filepath):
    """Parses thermal results for 0C depth (col 5) and Organism depth (col 7)."""
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found in the current directory.")
        return None

    try:
        df_raw = pd.read_csv(filepath, sep='\t', comment='#', header=None, engine='python')
        
        data = []
        for _, row in df_raw.iterrows():
            try:
                age_y = float(row[0])
                if age_y == 0:
                    continue
                    
                width = float(row[1])
                d0c_val = str(row[5]).strip()
                dorg_val = str(row[7]).strip()
                
                if any(x.lower() in ['nan', 't_dike'] for x in [d0c_val, dorg_val]):
                    continue
                    
                data.append([
                    age_y / 1000.0, # Age in ky
                    width,          # Width in m
                    float(d0c_val), # Depth 0C in m
                    float(dorg_val) # Depth Org in m
                ])
            except (ValueError, TypeError, IndexError):
                continue

    except Exception as e:
        print(f"Error reading file: {e}")
        return None

    if not data:
        print("Error: No valid numeric data points found.")
        return None

    df = pd.DataFrame(data, columns=['age_ky', 'width', 'depth0C', 'depthOrg'])
    
    # Handle duplicates by averaging
    df = df.groupby(['age_ky', 'width']).mean().reset_index()
    
    # Coordinate Jitter to fix Qhull precision errors
    np.random.seed(42)
    df['age_ky'] += np.random.uniform(-1e-8, 1e-8, size=len(df))
    df['width'] += np.random.uniform(-1e-8, 1e-8, size=len(df))
        
    return df

def main():
    # Setup Argument Parser
    parser = argparse.ArgumentParser(
        description="Generate a contour map of thermal evolution from simulation results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-l", "--linear", 
        action="store_true", 
        help="Use a linear scale for the X-axis (Dike Age) instead of logarithmic."
    )
    parser.add_argument(
        "-i", "--input", 
        default="thermal_evol.results.txt", 
        help="Path to the simulation results file."
    )
    args = parser.parse_args()

    df = load_and_clean_data(args.input)
    
    if df is None:
        return

    # Filter data to start at or after 1 ky
    df_plot = df[df['age_ky'] >= 1.0].copy()
    
    if df_plot.empty:
        print("Warning: No data points found with age >= 1 ky.")
        return

    # Create a dense grid based on the chosen scale
    if args.linear:
        xi_lin = np.linspace(1.0, df_plot['age_ky'].max(), 300)
    else:
        xi_lin = np.logspace(np.log10(1.0), np.log10(df_plot['age_ky'].max()), 400)
    
    yi_lin = np.linspace(df_plot['width'].min(), df_plot['width'].max(), 400)
    xi, yi = np.meshgrid(xi_lin, yi_lin)

    # Interpolation refinement
    try:
        zi_0c = griddata((df_plot['age_ky'], df_plot['width']), df_plot['depth0C'], (xi, yi), method='linear', rescale=True)
        zi_org = griddata((df_plot['age_ky'], df_plot['width']), df_plot['depthOrg'], (xi, yi), method='linear', rescale=True)
    except Exception as e:
        print(f"Interpolation error: {e}")
        return

    # Create custom Colormap: Green (0) -> Red (500) -> Black (4000)
    color_pts = [
        (0.0, (0, 0.8, 0)),    # Green at 0m
        (0.125, (1, 0, 0)),    # Red at 500m
        (1.0, (0.1, 0.1, 0.1)) # Black/Dark Gray at 4000m
    ]
    cmap_name = 'green_red_black'
    cm = LinearSegmentedColormap.from_list(cmap_name, color_pts, N=256)

    # Setup the plot
    plt.style.use('default') 
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Threshold for the "Surface" zone
    surface_threshold = 25.0

    # Unified levels for general contours
    # We exclude the threshold from the general set to handle its label separately
    unified_levels = np.unique(np.concatenate([
        np.arange(0, 501, 50), 
        np.arange(1000, 4501, 500)
    ]))
    unified_levels = unified_levels[unified_levels != surface_threshold]

    # 1. Surface zone: Fill the area between 0 and 25m depth with green
    ax.contourf(xi, yi, zi_0c, levels=[0, surface_threshold], colors=['#e6ffed'], alpha=0.8, zorder=1)
    
    # 2. Subtle plot of data points
    ax.scatter(df_plot['age_ky'], df_plot['width'], color='gray', s=3, alpha=0.15, zorder=2)

    # 3. Special label for the Surface contour (25m)
    # We use clabel with a dictionary format to write "Surface" on the line.
    cp_surface = ax.contour(xi, yi, zi_0c, levels=[surface_threshold], colors='darkgreen', linewidths=2.0, linestyles='solid', zorder=4)
    ax.clabel(cp_surface, inline=True, fontsize=10, fmt={surface_threshold: 'Surface'}, colors='darkgreen')

    # 4. Plot general 0C Isotherm Depth (Solid lines)
    cp0 = ax.contour(xi, yi, zi_0c, levels=unified_levels, cmap=cm, vmin=0, vmax=4000, linewidths=1.5, linestyles='solid', zorder=3)
    ax.clabel(cp0, inline=True, fontsize=8, fmt='%1.0f', colors='black')

    # 5. Plot Organism Depth (Dashed lines)
    # FIXED: Ensure levels are sorted to avoid ValueError: Contour levels must be increasing
    org_levels = np.sort(np.append(unified_levels, surface_threshold))
    cp_org = ax.contour(xi, yi, zi_org, levels=org_levels, cmap=cm, vmin=0, vmax=4000, linewidths=1.2, linestyles='dashed', zorder=4)
    ax.clabel(cp_org, inline=True, fontsize=8, fmt='%1.0f', colors='black')

    # --- GEOLOGICAL ANNOTATIONS ---
    
    # Cerberus F.: Semitransparent area
    ax.fill_between([50, 210], 16, 24, color='grey', alpha=0.3, zorder=5)
    ax.text((50 + 210) / 2, 23.5, 'Cerberus F.', color='grey', fontsize=10, 
            fontweight='bold', ha='center', va='top')

    # Elysium F.: Semitransparent area
    ax.axhspan(80, 120, color='grey', alpha=0.2, zorder=0)
    
    # Set X-axis scale
    if args.linear:
        ax.set_xscale('linear')
    else:
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))
        ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=15))

    # --- AXIS LIMITS ---
    ax.set_xlim(1.0, df_plot['age_ky'].max())
    ax.set_ylim(df_plot['width'].min(), df_plot['width'].max())

    # Position calculations for annotations
    xlims = ax.get_xlim()
    if not args.linear:
        # Geometric mean for center of log axis
        x_center = 10**( (np.log10(xlims[0]) + np.log10(xlims[1])) / 2 )
    else:
        # Arithmetic mean for center of linear axis
        x_center = (xlims[0] + xlims[1]) / 2

    # Label for Elysium F.
    ax.text(x_center, 118, 'Elysium F.', color='grey', alpha=0.8, 
            fontsize=12, fontweight='bold', ha='center', va='top', style='italic')

    # --- COLORBAR ---
    norm = plt.Normalize(vmin=0, vmax=4000)
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.ax.invert_yaxis()
    cbar.set_label('Depth (m)', rotation=270, labelpad=15)
    cbar.set_ticks([0, 500, 1000, 2000, 3000, 4000])

    # Final Axis Setup
    ax.set_title('Shallowest depth of the 0°C isotherm (solid) and the migrating organisms (dashed)', fontsize=14, pad=20)
    ax.set_xlabel('Time t (dike age, ky)', fontsize=12)
    ax.set_ylabel('Dike half-width W (m)', fontsize=12)
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    ax.grid(False)
    
    plt.tight_layout()
    output_image = "results.depth0C_age_width.png"
    plt.savefig(output_image, dpi=300, facecolor='white')
    print(f"Success! Plot generated and saved as '{output_image}'")
    plt.show()

if __name__ == "__main__":
    main()