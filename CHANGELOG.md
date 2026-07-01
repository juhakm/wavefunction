# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.0.3] - 2026-07-01

### Added
- Dynamic Basis Registry: Introduced the `Basis` abstraction layer, decoupling `Wavefunction` from its strict
  dependency on the standard periodic discrete Fourier transform. Users can now register custom alternative
  orthonormal bases via `Wavefunction.register_basis()`.
- Multi-Basis Minimum Search Engine: The `spectral_complexity` function now evaluates all registered bases
  simultaneously and selects the optimal compression path (the minimum complexity profile) for the target state.
- Boundary Edge Leakage Diagnostics: Integrated automated edge tracking that warns users when a chosen basis
  is suffering from artificial high-frequency inflation due to domain boundary discontinuities
  (e.g., non-periodic functions analyzed on a Fourier grid).
- Combinatorial Stirling Phase Cost Model: Implemented a mathematically rigorous phase tracking model
  (`phase_cost_model="stirling"`), deriving information constraints across multi-mode states via $\log_2(n_{\text{retained}})$.

### Changed
- Standardized Amplitude Normalization: Uniformly aligned basis projection math using an explicit $1/\sqrt{N}$ vector scaling normalization
  across all candidate bases, moving away from unscaled raw NumPy FFT array defaults.
- Streamlined String Representations: Rewrote `Wavefunction.__repr__` to output lazy metadata
  definitions (`fidelity_target`, `phase_cost_model`, and `bases_avail`) to eliminate heavy, recursive, or accidental
  complexity evaluations during background debugging routines.

### Fixed
- Solomonoff Floating-Point Underflow: Solved a critical boundary condition bug where uncentered wave packet constructors generated massive,
  artificial high-frequency spikes at the grid edges, inflating $C_s$ to the point where `solomonoff_weight()` abruptly
  underflowed down to an un-physical `0.0`.



## [0.0.2] - 2026-06-26

### Added
- Comprehensive `README.md` documentation.
- Example and usage demonstrations in `examples/demos.py`.
- Comprehensive suite of unit tests (courtesy of Gemini™).
- Updated `pyproject.toml` and `CITATION.cff` project metadata.


## [0.0.1] - 2026-06-26

### Added
- Initial release (forked from the `QBitwave` module).

