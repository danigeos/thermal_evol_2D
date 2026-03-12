# Numerical and Physical Formulation

This document details the physical and numerical formulation used in the `thermal_evol.py` script for modeling the 2D thermal evolution of the Martian subsurface.

## 1. Governing Equation

The simulation solves the two-dimensional heat conduction equation, which describes how temperature `T` changes over time `t` and space (`x`, `z`):

ρ * c_p * ∂T/∂t = ∇ ⋅ (k ⋅ ∇T) + Q

Where:
-   `ρ` is the bulk density (kg/m³).
-   `c_p` is the bulk specific heat capacity (J/kg·K).
-   `k` is the bulk thermal conductivity (W/m·K).
-   `∇` is the gradient operator.
-   `Q` is a source term, used here to model latent heat.

This can be simplified by using thermal diffusivity, `κ = k / (ρ * c_p)`:

∂T/∂t = ∇ ⋅ (κ ⋅ ∇T)

In Cartesian coordinates, this expands to:

∂T/∂t = ∂/∂x(κ * ∂T/∂x) + ∂/∂z(κ * ∂T/∂z)

The script assumes `κ` is locally constant for the matrix assembly at each time step but is updated based on temperature, making the overall problem non-linear.

### 1.1. Temperature-Dependent Thermal Properties

The model calculates the bulk thermal diffusivity (`κ`) for a porous medium composed of basalt saturated with a "filler" material (either water ice or liquid water). This is handled by the `get_martian_kappa` function.

The properties of both the basalt and the filler are temperature-dependent:

-   **Basalt:** The specific heat capacity of basalt (`c_p_basalt`) is a linear function of temperature, based on Whittington et al. (2009).
-   **Filler (Ice/Water):** A phase change occurs at 0°C (273.15 K).
    -   **Below 0°C (Ice):** The specific heat and thermal conductivity of the ice filler are both functions of temperature.
    -   **Above 0°C (Water):** The properties for liquid water are used, which are treated as constant.

The bulk properties of the composite material are then calculated as follows:

1.  **Bulk Density (`ρ_bulk`):** A weighted average based on the porosity (`phi`).
    `ρ_bulk = (1 - φ) * ρ_basalt + (φ * ρ_filler)`
2.  **Bulk Conductivity (`k_bulk`):** A geometric mean model is used.
    `k_bulk = k_basalt^(1 - φ) * k_filler^φ`
3.  **Bulk Specific Heat (`c_p_bulk`):** A mass-weighted average.
    `c_p_bulk = ((1 - φ) * ρ_basalt * c_p_basalt + φ * ρ_filler * c_p_filler) / ρ_bulk`

Finally, the thermal diffusivity `κ` is calculated from these bulk properties:

`κ = k_bulk / (ρ_bulk * c_p_bulk)`

## 2. Numerical Method

The equation is discretized using a finite-difference method on a 2D grid.

### Spatial Discretization
The model employs a **stretched grid**, where the spacing between nodes (`dx`, `dz`) is not uniform. This allows for higher resolution in areas of interest, such as near the surface and the magmatic intrusions. The grid is generated using a power law based on a `stretch_factor`.

The spatial derivatives are approximated using a central difference scheme, adapted for a non-uniform grid.

### Temporal Discretization
An **implicit backward Euler scheme** is used for the time derivative. This method is unconditionally stable, allowing for larger time steps than explicit methods. The discretized equation for each grid node `(i, j)` takes the form:

(T_new - T_old) / Δt = F(T_new)

Where `F` represents the discretized spatial derivatives evaluated at the *new* time `t + Δt`. This results in a large system of linear equations:

**A** ⋅ **T_new** = **T_old**

-   **T_new** is a vector of the unknown temperatures at the next time step.
-   **T_old** is a vector of the known temperatures at the current time step.
-   **A** is a sparse matrix containing coefficients derived from `Δt`, `κ`, `dx`, and `dz`.

The script builds this sparse matrix `A` (using `scipy.sparse.csc_matrix`) and solves the system efficiently using a direct solver (`scipy.sparse.linalg.factorized`).

## 3. Boundary and Initial Conditions

-   **Initial Condition:** The domain starts with a linear geothermal gradient: `T(x, z, t=0) = T_surface + gradT * z`.
-   **Top Boundary (z=0):** A constant temperature is maintained: `T(x, 0, t) = T_surface`.
-   **Bottom Boundary:** A constant heat flux is approximated by fixing the temperature gradient: `∂T/∂z = gradT`.
-   **Side Boundaries:** The side boundaries are treated as insulating (zero heat flux), which is a natural outcome of the finite-difference stencil at the edges.
-   **Magmatic Intrusion:** For the duration of the eruption (`t <= t_eruption`), constant high temperatures (`T_dike`, `T_colada`) are imposed on the grid nodes corresponding to the dike and lava flow.

## 4. Latent Heat of Fusion (Enthalpy Method)

The phase change of water ice is a major factor. This is handled using a source-term or "enthalpy" approach, which avoids the need to track the moving melt front explicitly.

1.  **Stall Temperature (`dT_stall_total`):** The latent heat of fusion (`L_fusion`) is converted into an equivalent temperature capacity. This is the amount of "thermal energy" (expressed in °C) a grid cell must absorb to fully melt, or release to fully freeze.

    `dT_stall_total = (L_fusion * ρ_water * porosity) / (ρ_bulk * c_p_bulk)`

2.  **Melt State:** A `melt_state` array tracks the fraction of water (from 0.0 for pure ice to 1.0 for pure water) in each cell.

3.  **Energy Correction:** After solving the diffusion equation for a time step (`T_pred`), the temperatures are corrected:
    -   If a cell's temperature rises above 0°C (`T_pred > 0`), the energy surplus is used to increase its `melt_state`. The cell's temperature is capped at 0°C until it is fully melted.
    -   If a cell's temperature drops below 0°C (`T_pred < 0`), the energy deficit is supplied by freezing water. The cell's temperature is held at 0°C until it is fully frozen.

This method correctly accounts for the energy absorbed and released during phase change, causing a characteristic temperature plateau at 0°C in the results.

## 5. Adaptive Timestep

To improve efficiency, the simulation uses an adaptive timestep `Δt`. The `Δt` is adjusted based on the maximum temperature change (`delta`) in a single step:

-   If `delta` is larger than a tolerance (`tol_high`), the step is considered inaccurate, and `Δt` is reduced (`shrink_factor`).
-   If `delta` is smaller than a tolerance (`tol_low`), the simulation is very stable, and `Δt` is increased (`grow_factor`) to speed up the simulation.

This ensures that the simulation proceeds rapidly during quasi-steady-state periods while taking smaller, more accurate steps during periods of rapid thermal change (like the initial eruption).
