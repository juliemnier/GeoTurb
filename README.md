# GeoTurb GPU-based Pseudo-Spectral Solver for 2D Geophysical Flows


This repository contains the **first version** of a pseudo-spectral solver for 2D geophysical flows, optimized to run on GPUs. The solver uses **[FourierFlows.jl](https://github.com/FourierFlows/FourierFlows.jl)** for grid management and FFT operations, and is greatly inspired by **[GeophysicalFlows.jl](https://github.com/FourierFlows/GeophysicalFlows.jl)**—though slightly uglier in its current state, and probably less memory-efficient as it is. Provides more flexibility and features related to my own specific research interests.

The implementation is a recent translation from my MATLAB pseudo-spectral code and represents a working prototype of the solver. **Use with care**, as this version is still under active development.

---

## Features

- **Runs on GPU**
- **Fourier Transformations**: Relies on FourierFlows.jl for grid and FFT-related operations. Could change in the future.
- **MATLAB-to-Julia Translation**: Converted from a fully tested MATLAB pseudo-spectral solver for 2D geophysical flows.

---

## Dependencies

This solver requires the following Julia packages:
- [FourierFlows.jl](https://github.com/FourierFlows/FourierFlows.jl)
  Install it using:
  ```julia
  using Pkg
  Pkg.add("FourierFlows")
