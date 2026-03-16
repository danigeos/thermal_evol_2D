#!/usr/bin/env python3
#For animation:
#magick -delay 100 *_0000.png -delay 10 *_????.png -loop 20 thermal_evol.anim.gif
#For running all the results file:
#rm thermal_evol.results.txt thermal_evol_????.png; for w in 100 230 5 80 50 170 20 140 200 120 65 35 10 27 42 15; do thermal_evol.py -W $w; done
#rm thermal_evol.results.txt thermal_evol_????.png; for m in 1 30 .03 .1 .3 3 10; do for w in 100 200 10 80 50 170 20 140 120 65 35 20; do thermal_evol.py -W $w --migration $m; done; done

import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless backend: no GUI windows
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
from scipy.sparse import csr_matrix, csc_matrix
import os, base64, glob, subprocess, shutil, sys
import argparse
try:
    from PIL import Image  # fallback for GIF if imageio is unavailable
except Exception:
    Image = None
try:
    import imageio.v2 as imageio  # for MP4 encoding via ffmpeg
except Exception:
    imageio = None
from scipy.sparse.linalg import spsolve, factorized
from scipy.ndimage import distance_transform_edt, zoom
from scipy.interpolate import RegularGridInterpolator, griddata
from scipy.spatial import cKDTree

"""
2D thermal diffusion solver (implicit) with adaptive time step and STRETCHED grid.
Includes Latent Heat via Enthalpy/Energy Budgeting (Source Term approach).
Outputs are synchronized with plot_every frequency.
"""

# ===================== PARAMETERS (Defaults) =====================
Lx = 6000.0   
Lz = 6000.0   
Nx, Nz = 301, 301
stretch_factor = 1.6 #How concentrated is resolution along the top and left boundaries; 1 means homogeneous; 1.8 works

T_dike = 1300.0     #C, temperature of the vertical dike
T_colada = 1250.0
T_surface = -58.0
gradT = 0.01       #Initial T gradient
YR = 3600*24*365    #

# Ground Properties (Global)
porosity = 0.2      # Volume fraction of pores >0.2 according to Li
L_fusion = 334000.0 # J/kg (Latent heat of water, set to 0 for no latent heat)

# --- THERMAL PROPERTY CALCULATOR ---
def get_martian_kappa(temp_c, phi=porosity):
    """
    Calculates thermal diffusivity (kappa) for fractured basalt saturated with water or ice.
    Works for both scalar temperatures and NumPy arrays.
    """
    temp_k = temp_c + 273.15
    rho_basalt, k_basalt = 2900.0, 2.7
    cp_basalt = 800.0 + 0.5 * (temp_k - 273.15) #Whittington 2009
    
    ice_mask = temp_k < 273.15
    rho_filler = np.where(ice_mask, 917.0, 1000.0)
    cp_filler = np.where(ice_mask, 2100.0 + 7.5 * (temp_k - 273.15), 4184.0)  #CLifford et al., 1993 JGR
    # Ice conductivity is temp-dependent; avoid division by zero
    k_filler = np.where(ice_mask, 2.2 * (273.15 / np.maximum(temp_k, 1e-3)), 0.6)
    
    rho_bulk = (1 - phi) * rho_basalt + (phi * rho_filler)
    k_bulk = (k_basalt ** (1 - phi)) * (k_filler ** phi)
    cp_bulk = ((1 - phi) * rho_basalt * cp_basalt + phi * rho_filler * cp_filler) / rho_bulk
    return k_bulk / (rho_bulk * cp_bulk)

# Global kappa used for matrix construction (standard basalt-ice mix)
kappa = get_martian_kappa(T_surface)

#GEOMETRY:
W, D = 100.0, 300.0    # width and depth of dike (m) 
L, H = 3000.0, 100.0 # Length and thickness of the colada flow (m)

# Temperature PROFILE position
x_profile_pos = 1500.0 # Position of the secondary temperature vertical profile (m)

#Timing
dt = 2*YR       #default dt (even for dynamic dt)
t_eruption = 10*YR #duration of the imposed T_dike and T_colada
tmax = 250e3*YR
plot_every = 1   #how many steps between numbered plots
save_frames = True        
image_format = "png"      
image_dpi = 200

adaptive_dt = True #True for adaptative time step
dt_min, dt_max = 1*YR, 40e3*YR
tol_high, tol_low = 1.0, 0.1 # Maximum allowable error threshold, Minimum error threshold (to trigger step growth)
shrink_factor, grow_factor, safety = 0.5, 1.3, 0.9 # Factor to decrease and increase step size on failure
smooth_factor = 0.15  # Smoothing factor for dt change. (ratio of dt change at each step)

#Migration of organisms
vel_migration = 1.0  # Migration rate of organisms (Set to 0.0 to disable calculations)
Tmin_life = -10      # Lower temperature limit for organism migration (C) and for plotting (<0 due to salinity). Use at least <-.1 to show the T plateau due to melting
marker_spacing = Lx/200     # Computational resolution for Lagrangian markers (m)
dot_spacing_plot = Lx/100   # Visualization spacing (constant dot density)

# ===================== CLI & SUMMARY =====================
def setup_cli():
    global Lx, Lz, Nx, Nz, kappa, T_dike, T_colada, T_surface, gradT, W, L, H, D
    global dt, t_eruption, tmax, plot_every, image_dpi, image_format, vel_migration, stretch_factor, marker_spacing, dot_spacing_plot, initial_temp_file

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--Lx", type=float, default=Lx)
    parser.add_argument("--Lz", type=float, default=Lz)
    parser.add_argument("--Nx", type=int, default=Nx)
    parser.add_argument("--Nz", type=int, default=Nz)
    parser.add_argument("--stretch", type=float, default=stretch_factor)
    parser.add_argument("--kappa", type=float, default=kappa)
    parser.add_argument("--T_dike", type=float, default=T_dike)
    parser.add_argument("--T_colada", type=float, default=T_colada)
    parser.add_argument("--T_surface", type=float, default=T_surface)
    parser.add_argument("--gradT", type=float, default=gradT)
    parser.add_argument("-W", "--width", type=float, default=W)
    parser.add_argument("-L", "--length", type=float, default=L)
    parser.add_argument("-H", "--thickness", type=float, default=H)
    parser.add_argument("-D", "--depth_dike", type=float, default=D, help="Depth of the top of the dike (m)")
    parser.add_argument("--tmax", type=float, default=tmax/YR)
    parser.add_argument("--dt", type=float, default=dt/YR)
    parser.add_argument("--migration", type=float, default=vel_migration)
    parser.add_argument("--marker_spacing", type=float, default=marker_spacing)
    parser.add_argument("--dot_spacing_plot", type=float, default=dot_spacing_plot)
    parser.add_argument("--initial_temp_file", type=str, default=None, help="Path to a text file with initial temperature field (x, z, T columns). Overrides T_surface and gradT for initialization.")
    
    args = parser.parse_args()
    Lx, Lz, Nx, Nz = args.Lx, args.Lz, args.Nx, args.Nz
    kappa, T_dike, T_colada, T_surface, gradT = args.kappa, args.T_dike, args.T_colada, args.T_surface, args.gradT
    W, L, H, D = args.width, args.length, args.thickness, args.depth_dike
    dt, tmax = args.dt * YR, args.tmax * YR
    vel_migration, stretch_factor = args.migration, args.stretch
    marker_spacing = args.marker_spacing
    dot_spacing_plot = args.dot_spacing_plot
    initial_temp_file = args.initial_temp_file

def print_summary():
    x_coords = Lx * (np.linspace(0, 1, Nx)**stretch_factor)
    z_coords = Lz * (np.linspace(0, 1, Nz)**stretch_factor)
    dxs, dzs = np.diff(x_coords), np.diff(z_coords)
    
    print("\n" + "="*60)
    print("           THERMAL EVOLUTION SIMULATION PARAMETERS")
    print("="*60)
    print(f"Domain:      {Lx}m x {Lz}m  ({Nx}x{Nz} nodes)")
    print(f"Grid:        Stretched. dx: [{dxs.min():.2f}, {dxs.max():.2f}]m, dz: [{dzs.min():.2f}, {dzs.max():.2f}]m")
    print(f"Geometry:    W={W:.0f} m, D={D} m, L={L} m, H={H} m")
    print(f"Migration:   {vel_migration:.1f} m/y")
    print("-" * 60)
    print(f"Time:        tmax={tmax/YR:.0f}y, dt={dt/YR:.1f}y, Migr={vel_migration} m/y")
    print(f"Diffusivity: 0°C: {get_martian_kappa(0.0):.2e} m^2/s; {T_surface}°C): {kappa:.2e} m^2/s")
    if initial_temp_file:
        print(f"Initial T:   From file '{initial_temp_file}'")
    else:
        print(f"Physics:     porosity={porosity:.2e}, gradT={gradT} C/m")
        print(f"Temps:       Surface={T_surface}C, Dike={T_dike}C, Flow={T_colada}C")
    print("="*60 + "\n")

setup_cli()
print_summary()

# --- Grid Definition ---
x = Lx * (np.linspace(0, 1, Nx)**stretch_factor)
z = Lz * (np.linspace(0, 1, Nz)**stretch_factor)

# --- Colormap ---
T_values = np.array([-60, -50, -40, -30, -20, -10, 0, 50, 100, 200, 400, 800, T_dike])
T_colors = ['#081d58', '#1d4f91', '#1f78b4', '#33a3c3', '#4ecdc4', '#7ad151', '#a5db36', '#fdae32', '#fd8d3c', '#f03b20', '#bd0026', '#800026', '#4d0018']
positions = (T_values - T_values[0]) / (T_values[-1] - T_values[0])
custom_cmap = LinearSegmentedColormap.from_list('temp_grad', list(zip(positions, T_colors)), N=512)

# --- Lagrangian Helpers ---
def calculate_populated_area(points, spacing):
    """Estimates populated area in m2 based on marker count."""
    if points.size == 0: return 0.0
    return points.shape[0] * (spacing ** 2)

def init_lagrangian_front(x_coords, z_coords, T_field):
    rgi = RegularGridInterpolator((z_coords, x_coords), T_field, bounds_error=False, fill_value=np.nan)
    xs = np.arange(0, Lx, marker_spacing)
    zs = np.arange(0, Lz, marker_spacing)
    zv, xv = np.meshgrid(zs, xs, indexing='ij')
    pts = np.column_stack((xv.ravel(), zv.ravel()))
    temps = rgi(np.column_stack((pts[:, 1], pts[:, 0])))
    valid = (temps >= Tmin_life) & (temps <= 100.0)
    return pts[valid]

def evolve_lagrangian_front(points, Tcur, x_coords, z_coords, v_mig, dt_s):
    """Updates population cloud with repulsion and prunning."""
    if points.size < 5: return points
    rgi = RegularGridInterpolator((z_coords, x_coords), Tcur, bounds_error=False, fill_value=-999)
    total_dist = v_mig * (dt_s / YR)
    if total_dist <= 0: return points
    step_dist_limit = marker_spacing * 1.5
    num_substeps = int(np.ceil(total_dist / step_dist_limit))
    step_dist = total_dist / num_substeps
    current_pop = points.copy()
    for _ in range(num_substeps):
        tree = cKDTree(current_pop)
        indices_list = tree.query_ball_point(current_pop, r=marker_spacing)
        surrounded_mask = np.zeros(len(current_pop), dtype=bool)
        for i, neighbors in enumerate(indices_list):
            if len(neighbors) < 9: continue
            diffs = current_pop[neighbors] - current_pop[i]
            mask = np.any(diffs != 0, axis=1)
            if not np.any(mask): continue
            valid_diffs = diffs[mask]
            angles = np.arctan2(valid_diffs[:, 1], valid_diffs[:, 0])
            sectors = ((angles + np.pi) / (2 * np.pi) * 8).astype(int) % 8
            if len(np.unique(sectors)) == 8: surrounded_mask[i] = True
        active_pop = current_pop[~surrounded_mask]
        if active_pop.size < 5: 
            current_pop = active_pop
            break
        num_points = active_pop.shape[0]
        k_neighbors = min(5, num_points) 
        
        tree_active = cKDTree(active_pop)
        dists, idxs = tree_active.query(active_pop, k=k_neighbors)
        
        repulsion_vec = np.zeros_like(active_pop)
        # Only iterate through the neighbors that were actually found
        for j in range(1, k_neighbors):
            d = dists[:, j]
            weight = 1.0 / np.maximum(d, 1e-6)
            neighbor_pts = active_pop[idxs[:, j]]
            diff = (active_pop - neighbor_pts)
            norm_diff = np.linalg.norm(diff, axis=1, keepdims=True)
            norm_diff[norm_diff == 0] = 1.0
            repulsion_vec += (diff / norm_diff) * weight[:, np.newaxis]
        r_norms = np.linalg.norm(repulsion_vec, axis=1, keepdims=True)
        r_norms[r_norms == 0] = 1.0
        jitter = (np.random.rand(*repulsion_vec.shape) - 0.5) * 0.05
        move_dir = (repulsion_vec / r_norms) + jitter
        buds = active_pop + move_dir * step_dist
        if step_dist > marker_spacing:
            inter = active_pop + move_dir * step_dist * 0.5
            all_cand = np.concatenate([buds, inter, active_pop], axis=0)
        else:
            all_cand = np.concatenate([buds, active_pop], axis=0)
        all_cand[:, 0] = np.clip(all_cand[:, 0], 0, Lx)
        all_cand[:, 1] = np.clip(all_cand[:, 1], 0, Lz)
        c_temps = rgi(np.column_stack((all_cand[:, 1], all_cand[:, 0])))
        all_cand = all_cand[(c_temps >= Tmin_life) & (c_temps <= 100.0)]
        if all_cand.shape[0] < 5: 
            current_pop = all_cand
            break
        tree_p = cKDTree(all_cand)
        indices_p = tree_p.query_ball_point(all_cand, r=marker_spacing * 0.75)
        keep = np.ones(len(all_cand), dtype=bool)
        for i, neighbors in enumerate(indices_p):
            if not keep[i]: continue
            for neighbor in neighbors:
                if neighbor > i: keep[neighbor] = False
        current_pop = all_cand[keep]
    return current_pop

# --- Diffusion Solver ---
def matriz_implicita_stretched(x, z, dt, kappa):
    Nx, Nz = len(x), len(z)
    N = Nx * Nz
    data, rows, cols = [], [], []
    dx, dz_diff = np.diff(x), np.diff(z)
    for i in range(Nz):
        for j in range(Nx):
            p = i * Nx + j
            center = 1.0
            if j == 0:
                hR = dx[j]
                coeff = 2.0 * kappa * dt / (hR**2)
                center += coeff
                rows.append(p); cols.append(p+1); data.append(-coeff)
            elif j == Nx - 1:
                hL = dx[j-1]
                coeff = 2.0 * kappa * dt / (hL**2)
                center += coeff
                rows.append(p); cols.append(p-1); data.append(-coeff)
            else:
                hL, hR = dx[j-1], dx[j]
                fac = 2.0 * kappa * dt / (hL + hR)
                wL, wR = fac / hL, fac / hR
                center += (wL + wR)
                rows.append(p); cols.append(p-1); data.append(-wL)
                rows.append(p); cols.append(p+1); data.append(-wR)
            if i == 0: pass
            elif i == Nz - 1:
                hL = dz_diff[i-1]
                coeff = 2.0 * kappa * dt / (hL**2)
                center += coeff
                rows.append(p); cols.append(p-Nx); data.append(-coeff)
            else:
                hL, hR = dz_diff[i-1], dz_diff[i]
                fac = 2.0 * kappa * dt / (hL + hR)
                wU, wD = fac / hL, fac / hR
                center += (wU + wD)
                rows.append(p); cols.append(p-Nx); data.append(-wU)
                rows.append(p); cols.append(p+Nx); data.append(-wD)
            rows.append(p); cols.append(p); data.append(center)
    return csc_matrix((data, (rows, cols)), shape=(N, N))

def _imponer_bcs_inplace(Tarr, t, x, z, T_dike, T_surface, t_eruption, W, L, H, D, gradT):
    Tarr[0, :] = T_surface
    if t <= t_eruption:
        nW, nL, nH = np.searchsorted(x, W), np.searchsorted(x, L), np.searchsorted(z, H)
        nD = np.searchsorted(z, D)
        Tarr[max(1, nD):, :nW] = T_dike
        Tarr[1:nH, :nL] = T_colada
    dz_last = z[-1] - z[-2]
    Tarr[-1, :] = Tarr[-2, :] + gradT * dz_last

def log_results(t_curr, T_field, lagrangian_points, z_coarse, W_val, migr_val, results_path):
    T_x0 = T_field[:, 0]
    idx0, idx100 = np.where(T_x0 >= 0.1)[0], np.where(T_x0 >= 100.0)[0]
    d0_x0, d100_x0 = (z_coarse[idx0[0]] if idx0.size > 0 else np.nan), (z_coarse[idx100[0]] if idx100.size > 0 else np.nan)
    mask0 = (T_field >= 0.1); has0 = mask0.any(axis=0)
    d0_min = np.min(z_coarse[np.argmax(mask0, axis=0)[has0]]) if has0.any() else np.nan
    mask100 = (T_field >= 100.0); has100 = mask100.any(axis=0)
    d100_min = np.min(z_coarse[np.argmax(mask100, axis=0)[has100]]) if has100.any() else np.nan
    d_org = np.min(lagrangian_points[:, 1]) if lagrangian_points.size > 0 else np.nan
    area_m2 = calculate_populated_area(lagrangian_points, marker_spacing)
    with open(results_path, "a") as f:
        f.write(f"{t_curr/YR:.2f}\t{W_val}\t{migr_val}\t{d0_x0:.2f}\t{d100_x0:.2f}\t{d0_min:.2f}\t{d100_min:.2f}\t{d_org:.2f}\n")
    return d_org, d0_min

def write_zT_profile(filepath, z, T_x0, T_xprof, time, step, x_prof_val):
    header = f"# THERMAL PROFILE DATA\n# Time: {time/YR:.2f} y\n# Position 1: x = 0.00 m\n# Position 2: x = {x_prof_val:.2f} m"
    data = np.column_stack((z, T_x0, T_xprof))
    np.savetxt(filepath, data, fmt="%.6e", header=header)

# ===================== SIMULATION CORE =====================
def simulacion(T_init, x, z, kappa, dt, tmax, T_dike, t_eruption, T_surface, gradT, adaptive_dt=True):
    Nx, Nz = len(x), len(z)
    Tcur = T_init.copy(); prev_dt=None; t=0.0; step=0; delta_prev = None

    # Energy Budget Setup; latent heat (Using global porosity and L_fusion)
    rho_water, cp_bulk, rho_bulk = 1000.0, 900.0, 2700.0
    
    # Calculate total stall temperature equivalent
    dT_stall_total = (L_fusion * rho_water * porosity) / (rho_bulk * cp_bulk)
    
    melt_state = np.zeros((Nz, Nx))
    melt_state[Tcur > 0] = 1.0 
    nW_init, nH_init, nD_init = np.searchsorted(x, W), np.searchsorted(z, H), np.searchsorted(z, D)
    melt_state[max(1, nD_init):, :nW_init] = 1.0
    melt_state[1:nH_init, :np.searchsorted(x, L)] = 1.0

    A = matriz_implicita_stretched(x, z, dt, kappa); solveA = factorized(A)
    results_path, profile_path = "thermal_evol.results.txt", "thermal.evol.zT"
    
    if not os.path.exists(results_path):
        with open(results_path, "w") as f:
            f.write("#time(y) width(m) migration_rate(m/y) depth0C_x0(m) depth100C_x0(m) depth0C_min(m) depth100C_min(m) depth_org_min(m)\n")

    # --- PLOTTING LAYOUT ---
    fig, (ax_map, ax_prof) = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={'width_ratios': [1, 1]})
    fig.subplots_adjust(wspace=0.13, left=0.15, right=0.92) 
    x_km, z_km = x / 1000.0, z / 1000.0
    Xm, Zm = np.meshgrid(x_km, z_km)
    asp = Lz / Lx
    ax_map.set_box_aspect(asp); ax_prof.set_box_aspect(asp)
    
    pc = ax_map.pcolormesh(Xm, Zm, Tcur, cmap=custom_cmap, vmin=T_values[0], vmax=T_values[-1], shading='auto')
    z_limit = z_km.max()
    ax_map.set_ylim(z_limit, 0); ax_map.set_xlabel('x (km)'); ax_map.set_ylabel('Depth (km)') 
    cax = inset_axes(ax_map, width="4%", height="100%", loc='lower left', bbox_to_anchor=(-0.25, 0, 1, 1), bbox_transform=ax_map.transAxes, borderpad=0)
    cbar = plt.colorbar(pc, cax=cax)
    cbar.ax.yaxis.set_ticks_position('left'); cbar.ax.yaxis.set_label_position('left')
    cbar.ax.invert_yaxis(); cbar.set_label('Temperature (°C)')
    
    ax_prof.set_ylim(z_limit, 0); ax_prof.set_xlabel('Temperature (°C)'); ax_prof.set_title('Profiles')
    ax_prof.set_ylabel('Depth (km)'); ax_prof.grid(True)
    
    # Twin Axis for Diffusivity
    ax_diff = ax_prof.twiny()
    ax_diff.set_box_aspect(asp)
    ax_diff.set_xlabel('Diffusivity ($\kappa$, m$^2$/s)', color='tab:gray', fontsize='small')
    ax_diff.tick_params(axis='x', labelcolor='tab:gray', labelsize='small')
    ax_diff.set_xlim(0.4e-6, 1.6e-6)
    
    j0, j_prof = 0, np.searchsorted(x, x_profile_pos)
    line_x0, = ax_prof.plot(Tcur[:, j0], z_km, label='T, x = 0.00 km', color='tab:blue')
    line_x1, = ax_prof.plot(Tcur[:, j_prof], z_km, label=f'T, x = {x_km[j_prof]:.2f} km', color='tab:orange')
    
    k_vec0, k_vec1 = get_martian_kappa(Tcur[:, j0]), get_martian_kappa(Tcur[:, j_prof])
    line_k0, = ax_diff.plot(k_vec0, z_km, color='tab:blue', linestyle='--', alpha=0.5, label='$\kappa$, x=0')
    line_k1, = ax_diff.plot(k_vec1, z_km, color='tab:orange', linestyle='--', alpha=0.5, label='$\kappa$, x=prof')
    ax_prof.plot(T_surface + gradT*z, z_km, 'k:', alpha=0.6, label='Initial (T)')
    ax_prof.legend(loc='lower right', fontsize='x-small')
    ax_prof.set_xlim(T_values[0], T_dike * 1.05)
    
    dots_artist = None; contour_artists = []; script_base = 'thermal_evol'
    T_initial_frame = T_init.copy()
    _imponer_bcs_inplace(T_initial_frame, 0, x, z, T_dike, T_surface, t_eruption, W, L, H, D, gradT)
    pc.set_array(T_initial_frame.flatten()) 
    
    if vel_migration > 0: pop_points = init_lagrangian_front(x, z, T_initial_frame)
    else: pop_points = np.array([])
        
    # --- Step 0 Outputs ---
    d_org, d0 = log_results(0.0, T_initial_frame, pop_points, z, W, vel_migration, results_path)
    print(f"step={0:6d}  0.0%  t=0.0y  dt={dt/YR:5.1f}y  d_org={d_org:7.1f}m  d_0.1C={d0:7.1f}m  pts={pop_points.shape[0]}")
    if save_frames:
        ax_map.set_title(f"t=0 y, W={int(W)}m, H={int(H)}m, L={int(L)}m, D={int(D)}m, Migr={vel_migration}m/y", fontsize=9, pad=15)
        fig.savefig(f"{script_base}_{0:04d}.{image_format}", dpi=image_dpi, bbox_inches='tight')
        write_zT_profile(profile_path, z, T_initial_frame[:, j0], T_initial_frame[:, j_prof], 0.0, 0, x[j_prof])

    while t < tmax - 1e-12:
        b = Tcur.flatten()
        dz_last = z[-1] - z[-2]
        rz_last = 2.0 * kappa * dt / (dz_last**2)
        b[np.arange((Nz-1)*Nx, Nz*Nx)] += rz_last * gradT * dz_last
        T_pred = solveA(b).reshape(Nz, Nx)
        
        # --- LATENT HEAT / ENTHALPY CORRECTION ---
        Tnew = T_pred.copy()
        if dT_stall_total > 1e-9:
            mask_melt = (Tcur <= 0) & (T_pred > 0)
            if np.any(mask_melt):
                energy_avail = T_pred[mask_melt]
                melt_needed = (1.0 - melt_state[mask_melt]) * dT_stall_total
                melt_applied = np.minimum(energy_avail, melt_needed)
                melt_state[mask_melt] += melt_applied / dT_stall_total
                Tnew[mask_melt] = T_pred[mask_melt] - melt_applied
                
            mask_freeze = (Tcur >= 0) & (T_pred < 0)
            if np.any(mask_freeze):
                energy_loss = -T_pred[mask_freeze]
                freeze_needed = melt_state[mask_freeze] * dT_stall_total
                freeze_applied = np.minimum(energy_loss, freeze_needed)
                melt_state[mask_freeze] -= freeze_applied / dT_stall_total
                Tnew[mask_freeze] = T_pred[mask_freeze] + freeze_applied
            Tnew[np.abs(Tnew) < 1e-10] = 0.0
            
        _imponer_bcs_inplace(Tnew, t, x, z, T_dike, T_surface, t_eruption, W, L, H, D, gradT)

        delta = float(np.max(np.abs(Tnew - Tcur)))
        if adaptive_dt:
            if delta_prev is not None:
                if delta < tol_low: dt *= grow_factor
                elif delta/(delta_prev + 1e-12) < 1.1: dt *= (1 + smooth_factor)
                elif delta > tol_high: dt = max(dt_min, safety*dt * shrink_factor)
            dt = np.clip(dt, dt_min, dt_max)
        if t + dt > tmax: dt = max(dt_min, tmax - t)
        if prev_dt != dt:
            A = matriz_implicita_stretched(x, z, dt, kappa); solveA = factorized(A); prev_dt = dt

        Tcur = Tnew.copy(); t += dt; step += 1; delta_prev = delta
        if vel_migration > 0: pop_points = evolve_lagrangian_front(pop_points, Tcur, x, z, vel_migration, dt)
        else: pop_points = np.array([])

        # --- SYNCHRONIZED OUTPUTS ---
        if step % plot_every == 0:
            # 1. Update Graphics
            pc.set_array(Tcur.flatten())
            for c in contour_artists:
                try: c.remove()
                except: pass
            
            # Plot main 100C and Tmin_life contours as solid, thick lines
            # Explicitly setting linestyles='solid' prevents negative contours (Tmin_life = -2.0) from being dashed
            c_main = ax_map.contour(Xm, Zm, Tcur, levels=[Tmin_life, 100.0], colors='white', linewidths=1.5, linestyles='solid')
            # Plot 0.1C as a dashed, thinner line
            c_phase = ax_map.contour(Xm, Zm, Tcur, levels=[0.1], colors='white', linewidths=0.8, linestyles='dashed')
            contour_artists = [c_main, c_phase]
            
            line_x0.set_xdata(Tcur[:, j0]); line_x1.set_xdata(Tcur[:, j_prof])
            line_k0.set_xdata(get_martian_kappa(Tcur[:, j0])); line_k1.set_xdata(get_martian_kappa(Tcur[:, j_prof]))
            ax_map.set_title(f"t={int(t/YR)} y, W={int(W)}m, H={int(H)}m, L={int(L)}m, D={int(D)}m, Migr={vel_migration}m/y", fontsize=9, pad=15)
                
            # --- FIX: Safely remove the previous dots artist ---
            if dots_artist is not None:
                try:
                    dots_artist.remove()
                except (ValueError, RuntimeError):
                    pass
                dots_artist = None # Reset to ensure we don't try removing it again

            if pop_points.size > 0:
                vis_xs, vis_zs = np.arange(0, Lx, dot_spacing_plot), np.arange(0, Lz, dot_spacing_plot)
                vzv, vxv = np.meshgrid(vis_zs, vis_xs, indexing='ij')
                vis_grid = np.column_stack((vxv.ravel(), vzv.ravel()))
                tree_v = cKDTree(pop_points)
                dist, _ = tree_v.query(vis_grid, k=1, distance_upper_bound=dot_spacing_plot * 0.95)
                to_plot = vis_grid[np.isfinite(dist)]
                if to_plot.size > 0:
                    rgi_vis = RegularGridInterpolator((z, x), Tcur, bounds_error=False, fill_value=-999)
                    habitat_mask = (rgi_vis(np.column_stack((to_plot[:, 1], to_plot[:, 0]))) >= Tmin_life) & (rgi_vis(np.column_stack((to_plot[:, 1], to_plot[:, 0]))) <= 100.0)
                    to_plot = to_plot[habitat_mask]
                    if to_plot.size > 0: 
                        # --- FIX: Assign the new scatter artist ---
                        dots_artist = ax_map.scatter(to_plot[:, 0]/1000.0, to_plot[:, 1]/1000.0, s=.9, c='k', alpha=0.8, linewidths=0)

            # 2. Log to file, print to screen, and write zT profile
            d_org, d0 = log_results(t, Tcur, pop_points, z, W, vel_migration, results_path)
            print(f"step={step:6d}  t={t/YR:7.1f}y  dt={dt/YR:5.1f}y  d_org={d_org:7.1f}m  d_0.1C={d0:7.1f}m  pts={pop_points.shape[0]}")
            
            if save_frames: 
                fig.savefig(f"{script_base}_{step:04d}.{image_format}", dpi=image_dpi, bbox_inches='tight')
                write_zT_profile(profile_path, z, Tcur[:, j0], Tcur[:, j_prof], t, step, x[j_prof])
                
    plt.close(fig)

if __name__ == "__main__":
    # Initialize with background geotherm, which serves as a default or a base
    T_init_field = np.zeros((Nz, Nx))
    for i in range(Nz): T_init_field[i, :] = T_surface + gradT * z[i]

    # If an initial temperature file is provided, load and interpolate it onto the grid
    if initial_temp_file:
        print(f"Loading and interpolating initial temperature from '{initial_temp_file}'...")
        try:
            # Load data (x, z, T)
            data = np.loadtxt(initial_temp_file, comments='#')
            points = data[:, :2]  # x, z
            values = data[:, 2]   # T
            
            # Create the grid to interpolate onto
            X_grid, Z_grid = np.meshgrid(x, z)
            
            # Interpolate. 'linear' is robust. NaNs are returned for points outside the convex hull.
            T_interp = griddata(points, values, (X_grid, Z_grid), method='linear')
            
            # Where T_interp is not NaN, use it. Otherwise, keep the background geotherm.
            T_init_field = np.where(np.isnan(T_interp), T_init_field, T_interp)
            print("Successfully applied initial temperature field from file.")
        except Exception as e:
            print(f"FATAL: Could not load or process initial temperature file: {e}. Aborting.")
            sys.exit(1)

    simulacion(T_init_field, x, z, kappa, dt, tmax, T_dike, t_eruption, T_surface, gradT, adaptive_dt)