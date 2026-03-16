#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata

def load_simulation_data(filepath):
    """
    Parses thermal_evol.results.txt.
    Target Columns: 
    1: Width (W)
    2: Migration Rate (Migr)
    7: depth_org_min (Depth)
    """
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        return None

    try:
        # Load data, skipping comments
        df_raw = pd.read_csv(filepath, sep='\t', comment='#', header=None, engine='python')
        
        # Column mapping based on user description:
        # 0:time, 1:width, 2:migr, 3:d0C_x0, 4:d100C_x0, 5:d0C_min, 6:d100C_min, 7:d_org
        df_raw.columns = ['time', 'width', 'migr', 'd0_x0', 'd100_x0', 'd0_min', 'd100_min', 'depth_org']

        # Clean non-numeric or infinite values
        df_raw = df_raw.replace([np.inf, -np.inf], np.nan).dropna(subset=['width', 'migr', 'depth_org'])
        
        # For each (width, migration) pair, find the absolute shallowest depth reached across all time
        # This represents the "Maximum colonization potential" for that dike geometry
        df_summary = df_raw.groupby(['width', 'migr'])['depth_org'].min().reset_index()

        return df_summary

    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def main():
    input_file = "thermal_evol.results.txt"
    df = load_simulation_data(input_file)
    
    if df is None or df.empty:
        print("No valid data to plot.")
        return

    # Create dense grid for interpolation
    # Note: Migration rate is often plotted on a log scale if it spans orders of magnitude
    xi_lin = np.linspace(df['migr'].min(), df['migr'].max(), 300)
    yi_lin = np.linspace(df['width'].min(), df['width'].max(), 300)
    xi, yi = np.meshgrid(xi_lin, yi_lin)

    # Interpolate Organism Depth
    zi = griddata((df['migr'], df['width']), df['depth_org'], (xi, yi), method='linear')

    # Custom Colormap: Shallow (Green) -> Mid (Red) -> Deep (Black)
    color_pts = [
        (0.0, (0, 0.8, 0)),    # Green (Shallow)
        (0.125, (1, 0, 0)),    # Red (~500m)
        (1.0, (0.1, 0.1, 0.1)) # Dark Gray/Black (Deep)
    ]
    cm = LinearSegmentedColormap.from_list('org_depth_cmap', color_pts, N=256)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor('white')

    # Define contour levels (adjust based on your expected Lz)
    levels = np.unique(np.concatenate([
        [25], # Surface threshold
        np.arange(0, 501, 50),
        np.arange(1000, 5001, 500)
    ]))

    # Fill background for the "Surface" zone (< 25m)
    ax.contourf(xi, yi, zi, levels=[0, 25], colors=['#e6ffed'], alpha=0.8)

    # Plot primary contours
    cp = ax.contour(xi, yi, zi, levels=levels, cmap=cm, vmin=0, vmax=4000, linewidths=1.5)
    ax.clabel(cp, inline=True, fontsize=9, fmt='%1.0f', colors='black')

    # Scatter actual data points to show coverage
    ax.scatter(df['migr'], df['width'], color='black', s=10, alpha=0.3, label='Simulated Dikes')

    # Axis Formatting
    ax.set_title('Minimum Depth of Organisms reached during Dike Cooling', fontsize=14, pad=15)
    ax.set_xlabel('Migration Rate (m/y)', fontsize=12)
    ax.set_ylabel('Dike Width W (m)', fontsize=12)
    
    # Toggle log scale for migration if needed
    # ax.set_xscale('log') 

    # Colorbar
    norm = plt.Normalize(vmin=0, vmax=4000)
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.ax.invert_yaxis()
    cbar.set_label('Shallowest Organism Depth (m)', rotation=270, labelpad=15)

    plt.tight_layout()
    plt.savefig("org_depth_migration_width.png", dpi=300)
    print("Plot saved as 'org_depth_migration_width.png'")
    plt.show()

if __name__ == "__main__":
    main()