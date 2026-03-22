# 2D Thermal Evolution Model for Mars

This project simulates the 2D thermal evolution of the Martian subsurface following the emplacement of a magmatic dike and a subsequent lava flow ("colada"). It is written in Python and uses a finite-difference method on a stretched grid with an implicit time-stepping scheme.

## Key Features

-   **2D Thermal Diffusion:** Solves the heat diffusion equation in two dimensions.
-   **Martian Environment:** Uses thermal properties specific to Martian materials (basalt, water ice).
-   **Latent Heat:** Incorporates the latent heat of fusion/melting for water ice using an enthalpy method.
-   **Complex Geometry:** Models a vertical dike and a horizontal lava flow.
-   **Adaptive Time-Stepping:** Adjusts the time step dynamically to ensure stability and efficiency.
-   **Stretched Grid:** Provides higher spatial resolution near the surface and the dike.
-   **Organism Migration:** Includes a Lagrangian particle simulation to track the potential habitable zone for microbial life based on temperature (`-10°C` to `100°C`).
-   **Highly Configurable:** Simulation parameters can be easily adjusted via command-line arguments.

## Dependencies

The script requires several common scientific Python libraries. You can install them using pip:

```bash
pip install numpy matplotlib scipy pillow imageio
```

The script also uses `magick` (from ImageMagick) for creating animations. This is an optional dependency that needs to be installed separately.

## How to Run

The simulation can be run directly from the command line.

**Basic Execution:**

```bash
python3 thermal_evol.py
```

This will run the simulation with the default parameters defined at the top of the script.

**Custom Parameters:**

You can customize the simulation using a wide range of command-line arguments. For example, to change the width of the dike:

```bash
python3 thermal_evol.py -W 200
```

To see all available options and their default values, run:

```bash
python3 thermal_evol.py --help
```

The script includes comments at the top with examples for running simulations in a loop to test different parameters.

** Combine with a ¡n initial state of a 400-m meteorite impact **

```bash
meteorite_temperature_effect.py -D 400
thermal_evol.py --initial_temp_file meteorite_temperature_effect.txt -W 0 -H 0
```


## Output Files

The script generates several output files:

-   `thermal_evol_????.png`: A sequence of PNG images showing the 2D temperature field at different time steps.
-   `thermal_evol.results.txt`: A text file logging the evolution of key metrics over time, such as the depth of the 0°C and 100°C isotherms.
-   `thermal.evol.zT`: A data file containing the vertical temperature profiles at two different x-locations (x=0 and a user-defined position).

## Creating an Animation

The output PNG frames can be compiled into a GIF animation using ImageMagick's `magick` command. The command below is provided in the script's comments:

```bash
magick -delay 100 *_0000.png -delay 10 *_????.png -loop 20 thermal_evol.anim.gif
```
