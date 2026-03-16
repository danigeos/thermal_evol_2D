#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd
import re
import os

def plot_geothermal_data(filename):
    if not os.path.exists(filename):
        print(f"Error: {filename} not found in the current directory.")
        return

    try:
        with open(filename, 'r') as f:
            lines = f.readlines()

        data_tokens = []
        headers = []

        for line in lines:
            # 1. Strip comments and whitespace
            clean_line = line.split('#')[0].strip()
            
            # 2. Extract headers from the comment line if it contains the keywords
            if line.startswith('#') and 't(ky)' in line:
                # Remove the # and split into column names
                headers = line.replace('#', '').strip().split()

            # 3. Remove [source: X] tags using a safe regex pattern
            clean_line = re.sub(r'\[source:\s*\d+\]', '', clean_line)
            
            # 4. Collect numbers (tokens)
            if clean_line:
                data_tokens.extend(clean_line.split())

        # Check if we found headers
        if not headers:
            headers = ['t(ky)', 'z0C_W=100', 'z_migr_W=100', 'z0C_W=50', 'z_migr_W=50', 'z0C_W=25', 'z_migr_W=25']

        # 5. Group tokens into rows of 7
        rows = []
        for i in range(0, len(data_tokens), 7):
            row = data_tokens[i:i+7]
            if len(row) == 7:
                rows.append(row)

        # Create DataFrame
        df = pd.DataFrame(rows, columns=headers)
        # Convert all columns to numeric values
        df = df.apply(pd.to_numeric, errors='coerce')

        # 6. Plotting
        plt.figure(figsize=(10, 7))
        
        # Color mapping for W values
        color_map = {
            '100': '#1f77b4', # Blue
            '50': '#2ca02c',  # Green
            '25': '#d62728'   # Red
        }

        for w in ['100', '50', '25']:
            z0c_col = f'z0C_W={w}'
            z_migr_col = f'z_migr_W={w}'
            color = color_map[w]

            # Plot z0C (Plain Solid Line)
            if z0c_col in df.columns:
                plt.plot(df['t(ky)'], df[z0c_col], 
                         label=f'z0C (W={w})', 
                         color=color, 
                         linestyle='-', 
                         marker='o', 
                         markersize=4,
                         linewidth=2)

            # Plot z_migr (Dotted Line)
            if z_migr_col in df.columns:
                plt.plot(df['t(ky)'], df[z_migr_col], 
                         label=f'z_migr (W={w})', 
                         color=color, 
                         linestyle=':', 
                         marker='x', 
                         markersize=5,
                         linewidth=2)

        # Scientific Chart Styling
        plt.title('Shallowest depth of the 0°C isotherm and migration depth', fontsize=14, pad=15)
        plt.xlabel('Time (ky)', fontsize=12)
        plt.ylabel('Depth (km)', fontsize=12)
        
        # Invert Y-axis so 0 (surface) is at the top
        plt.gca().invert_yaxis()
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
        plt.tight_layout()

        # Save and Show
        output_file = 'results_z0C+migratCurves.png'
        plt.savefig(output_file, dpi=300)
        print(f"Success! Plot saved as {output_file}")
        plt.show()

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    plot_geothermal_data('results_z0C+migratCurves.txt')