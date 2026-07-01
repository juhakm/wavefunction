"""Spectral Complexity measure for complex-valued wavefunctions.

Theory background (Meskanen 2026 — "The Wavefunction as Compression")
======================================================================
The central hypothesis is that the quantum wavefunction is the universe's
data-compression codec. Internal observers — themselves composed of
compressed structures — perceive their constituent degrees of freedom as
wave-like because they are observing *compressed information*. The codec
that produces this compression is the spectral decomposition.

Spectral Complexity C_s
-----------------------
A wavefunction ψ(x) can always be written as a superposition of spectral
modes, each characterised by two attributes:

    frequency  ω  — the rate of oscillation, unbounded above zero
    phase      φ  — the offset of the oscillation, bounded in [0, 2π)

The *spectral complexity* C_s(ψ) is the total continuous information cost
needed to specify the set of modes that materially compose ψ:

    C_s(ψ) = Σ_i  [ |ω_i| / Δω  +  φ_cost(i) ]

Frequency cost (dominant term)
    |ω_i| / Δω is the number of resolution steps Δω needed to locate
    frequency ω_i. It is unbounded, continuous, and grows linearly with
    frequency. This term *dominates* C_s and is the reason the measure
    exponentially suppresses high-frequency (rough, chaotic) states.
    The identification Δω = ℏ ln 2 connects the minimum frequency
    resolution to Planck's constant (Meskanen 2026, §3.1).

Phase cost (subdominant, bounded)
    Each phase φ_i ∈ [0, 2π) requires a finite amount of information to
    specify. The cost is *global* over all modes. 
    
    Fix v2.00: The actual information needed to specify an unambiguous
    assignment of phases to n modes is a combinatorial cost that grows with n.
    Distinguishing n items requires about log2(n!) ~ n*log2(n) bits total 
    (Stirling). Thus, the default 'stirling' model assigns a marginal phase 
    cost per non-reference mode = log2(n_retained). The 'flat' proxy is 
    retained for backwards compatibility.

Amplitude and the fidelity engine
    Amplitude determines *which modes are included* in the description via a
    power-ranked fidelity engine: modes are added in descending power
    order until the accumulated power reaches a target fraction of the
    total. Modes below this threshold are simply absent from the
    description.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Basis abstraction (Integrated from spectral_v2.py)
# ---------------------------------------------------------------------------

@dataclass
class Basis:
    """A finite, orthonormal (on the sample grid) set of real or complex
    basis functions, plus the physical frequency associated with each.
    """
    name: str
    vectors: np.ndarray
    omega: np.ndarray
    delta_omega: float
    provenance: str = "generic"

    def project(self, psi: np.ndarray) -> np.ndarray:
        """Coefficients of psi in this basis via inner product."""
        return self.vectors.conj() @ psi

    @classmethod
    def fourier(cls, N: int, dx: float = 1.0) -> "Basis":
        """Standard periodic DFT basis (v0.01's only option)."""
        omega = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
        k = np.arange(N)
        vectors = np.exp(-2j * np.pi * np.outer(k, k) / N) / np.sqrt(N)
        delta_omega = 2.0 * np.pi / (N * dx)
        return cls("fourier", vectors, omega, delta_omega,
                   provenance="generic: assumes periodic boundary conditions")

    @classmethod
    def from_functions(
        cls,
        name: str,
        fn: Callable[[int, np.ndarray], np.ndarray],
        omega: np.ndarray,
        grid: np.ndarray,
        delta_omega: float,
        provenance: str = "generic",
    ) -> "Basis":
        """Build & orthonormalise (Gram-Schmidt) a basis from a generator fn."""
        n_modes = len(omega)
        raw = np.array([fn(k, grid) for k in range(n_modes)], dtype=complex)
        # Gram-Schmidt orthonormalisation (numerically robust via QR)
        Q, _ = np.linalg.qr(raw.T)
        vectors = Q.T[:n_modes]
        return cls(name, vectors, np.asarray(omega), delta_omega, provenance)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ComplexityResult:
    C_s: float
    basis_name: str
    n_modes_retained: int
    freq_cost: float
    phase_cost: float
    provenance: str
    all_basis_costs: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"ComplexityResult(C_s={self.C_s:.4f}, basis='{self.basis_name}', "
                f"modes={self.n_modes_retained}, freq={self.freq_cost:.4f}, "
                f"phase={self.phase_cost:.4f})")


# ---------------------------------------------------------------------------
# Spectral mode
# ---------------------------------------------------------------------------

@dataclass
class SpectralMode:
    """A single mode in the spectral decomposition of a wavefunction."""
    frequency: float
    amplitude: float
    phase: float

    def __post_init__(self) -> None:
        self.amplitude = float(abs(self.amplitude))
        self.phase = float(self.phase % (2.0 * np.pi))


# ---------------------------------------------------------------------------
# Wavefunction
# ---------------------------------------------------------------------------

class Wavefunction:
    """A normalised complex wavefunction on a uniform 1-D spatial grid."""

    def __init__(
        self,
        psi: np.ndarray,
        dx: float = 1.0,
        delta_omega: Optional[float] = None,
        fidelity_target: float = 0.999,
        phase_cost_model: str = "stirling",
        phase_resolution: float = 1.0,
    ) -> None:
        self._psi: np.ndarray = self._normalise(np.asarray(psi, dtype=complex))
        self.dx: float = float(dx)
        self.fidelity_target: float = float(fidelity_target)
        self.phase_cost_model: str = phase_cost_model
        self.phase_resolution: float = float(phase_resolution)

        N = len(self._psi)
        # Track bases library
        self.bases: dict[str, Basis] = {}
        
        # Calculate/Assign natural resolution constraints
        fourier_delta = 2.0 * np.pi / (N * self.dx)
        self.delta_omega: float = float(delta_omega) if delta_omega is not None else fourier_delta
        self.hbar_identified: float = self.delta_omega / np.log(2.0)

        # Register default fourier basis automatically
        # Note: dynamic bases overwrite delta_omega using their own configurations.
        self.register_basis(Basis.fourier(N, dx=self.dx))

    def register_basis(self, basis: Basis) -> None:
        """Registers a new Basis candidate library for robust searching."""
        if basis.vectors.shape[1] != self.N:
            raise ValueError(f"Basis size mismatch. Expected {self.N}, got {basis.vectors.shape[1]}")
        self.bases[basis.name] = basis

    # ------------------------------------------------------------------
    # Core properties
    # ------------------------------------------------------------------

    @property
    def psi(self) -> np.ndarray:
        """Normalised complex wavefunction array ψ(x), shape (N,)."""
        return self._psi

    @property
    def N(self) -> int:
        """Number of spatial grid points."""
        return len(self._psi)

    @property
    def x(self) -> np.ndarray:
        """Spatial grid positions x_n = n · dx, shape (N,)."""
        return np.arange(self.N) * self.dx

    @property
    def probability_density(self) -> np.ndarray:
        """|ψ(x)|² — Born-rule probability density, shape (N,)."""
        return np.abs(self._psi) ** 2

    # ------------------------------------------------------------------
    # Fidelity engine — mode selection
    # ------------------------------------------------------------------

    def retained_modes(self, basis_name: str = "fourier") -> list[SpectralMode]:
        """Return the set of spectral modes retained from a specified basis library.
        
        Args:
            basis_name: The target registered basis to query. Defaults to 'fourier'.
        """
        if basis_name not in self.bases:
            raise ValueError(f"Basis '{basis_name}' is not registered.")
        
        basis = self.bases[basis_name]
        coeffs = basis.project(self._psi)
        power = np.abs(coeffs) ** 2
        total_power = float(power.sum())

        if total_power == 0.0:
            return []

        sorted_idx = np.argsort(power)[::-1]
        accumulated = 0.0
        kept = []
        for idx in sorted_idx:
            accumulated += float(power[idx])
            kept.append(int(idx))
            if accumulated / total_power >= self.fidelity_target:
                break

        modes = [
            SpectralMode(
                frequency=float(basis.omega[k]),
                amplitude=float(np.abs(coeffs[k])) / self.N,
                phase=float(np.angle(coeffs[k]) % (2.0 * np.pi)),
            )
            for k in kept
        ]
        modes.sort(key=lambda m: abs(m.frequency))
        return modes

    # ------------------------------------------------------------------
    # Robust multi-basis spectral complexity C_s(ψ)
    # ------------------------------------------------------------------

    def spectral_complexity(
        self, 
        verbose: bool = False,
        edge_leakage_check: bool = True,
        edge_leakage_threshold: float = 0.05,
    ) -> float:
        """Compute the robust multi-basis spectral complexity C_s(ψ).

        Evaluates the code description cost across all registered bases and selects 
        the minimum viable configuration profile.
        """
        if not self.bases:
            raise ValueError("No bases registered.")

        all_costs: dict[str, float] = {}
        best_res: Optional[ComplexityResult] = None
        winning_modes: list[SpectralMode] = []
        winning_ref_indicators: list[bool] = []

        for basis in self.bases.values():
            coeffs = basis.project(self._psi)
            power = np.abs(coeffs) ** 2
            total_power = power.sum()
            if total_power == 0:
                continue

            order = np.argsort(power)[::-1]
            cum = 0.0
            kept = []
            for idx in order:
                cum += power[idx]
                kept.append(idx)
                if cum / total_power >= self.fidelity_target:
                    break

            n_kept = len(kept)
            freq_cost = float(np.sum(np.abs(basis.omega[kept]) / basis.delta_omega))

            if self.phase_cost_model == "stirling":
                per_mode = np.log2(n_kept) if n_kept > 1 else 0.0
                phase_cost = per_mode * max(n_kept - 1, 0)
            elif self.phase_cost_model == "flat":
                phase_cost = self.phase_resolution * max(n_kept - 1, 0)
            else:
                raise ValueError(f"unknown phase_cost_model: {self.phase_cost_model}")

            cs = freq_cost + phase_cost
            all_costs[basis.name] = cs

            # Construct temporary structures to align with your verbose printing loops
            current_modes = [
                SpectralMode(
                    frequency=float(basis.omega[k]),
                    amplitude=float(np.abs(coeffs[k])) / self.N,
                    phase=float(np.angle(coeffs[k]) % (2.0 * np.pi)),
                )
                for k in kept
            ]
            current_modes.sort(key=lambda m: abs(m.frequency))

            if best_res is None or cs < best_res.C_s:
                best_res = ComplexityResult(
                    C_s=cs, basis_name=basis.name, n_modes_retained=n_kept,
                    freq_cost=freq_cost, phase_cost=phase_cost,
                    provenance=basis.provenance,
                )
                winning_modes = current_modes
                
                # Determine reference index for localized phase metrics
                ref_amp = max(m.amplitude for m in winning_modes) if winning_modes else 0.0
                ref_f_abs = min(abs(m.frequency) for m in winning_modes if m.amplitude == ref_amp) if winning_modes else 0.0
                winning_ref_indicators = [
                    (m.amplitude == ref_amp and abs(m.frequency) == ref_f_abs) for m in winning_modes
                ]

        if best_res is None:
            raise ValueError("psi has zero power in every candidate basis")

        best_res.all_basis_costs = all_costs

        if verbose:
            print(f"\nWinning Basis: '{best_res.basis_name}' ({best_res.provenance})")
            print(f"  {'i':>4}  {'ω':>10}  {'A':>9}  {'φ/2π':>7}  "
                  f"{'|ω|/Δω':>10}  {'φ_cost':>7}  {'mode C_s':>9}")
            print("  " + "-" * 62)
            
            basis_obj = self.bases[best_res.basis_name]
            for i, mode in enumerate(winning_modes):
                f_cost = abs(mode.frequency) / basis_obj.delta_omega
                is_ref = winning_ref_indicators[i]
                
                # Deduce printed analytical slice step costs
                if self.phase_cost_model == "stirling":
                    p_cost = 0.0 if is_ref else np.log2(best_res.n_modes_retained)
                else:
                    p_cost = 0.0 if is_ref else self.phase_resolution
                
                m_cs = f_cost + p_cost
                print(f"  {i:>4}  {mode.frequency:>10.4f}  {mode.amplitude:>9.5f}"
                      f"  {mode.phase / (2*np.pi):>7.4f}"
                      f"  {f_cost:>10.3f}  {p_cost:>7.2f}  {m_cs:>9.3f}"
                      + ("  ← ref" if is_ref else ""))
            print("  " + "-" * 62)
            print(f"  C_s = {best_res.C_s:.4f}  (Freq: {best_res.freq_cost:.2f}, Phase: {best_res.phase_cost:.2f})")
            print(f"  All evaluated candidate options: {all_costs}\n")

        if edge_leakage_check and best_res.basis_name == "fourier":
            mag = np.abs(self._psi)
            edge = max(mag[0], mag[-1])
            peak = mag.max() + 1e-30
            if edge / peak > edge_leakage_threshold:
                print(f"  [warning] Fourier basis won, but |psi| at domain edge "
                      f"is {100*edge/peak:.1f}% of peak — signal is likely "
                      f"non-periodic on this grid; C_s may be leakage-inflated. "
                      f"Consider registering a boundary-matched basis.")

        return best_res.C_s

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    def solomonoff_weight(self) -> float:
        """Unnormalised Solomonoff prior weight 2^{-C_s(ψ)}."""
        return 2.0 ** (-self.spectral_complexity())

    def mode_suppression_factors(self, basis_name: str = "fourier") -> dict[float, float]:
        """Per-mode Boltzmann suppression factors from the frequency cost."""
        basis = self.bases[basis_name]
        return {
            m.frequency: 2.0 ** (-abs(m.frequency) / basis.delta_omega)
            for m in self.retained_modes(basis_name=basis_name)
        }

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def gaussian_packet(
        cls,
        N: int = 256,
        x0: float = 0.0,
        sigma: float = 1.0,
        k0: float = 1.0,
        dx: float = 0.1,
        **kwargs,
    ) -> "Wavefunction":
        """Construct a Gaussian wave packet."""
        x = np.arange(N) * dx
        psi = np.exp(-((x - x0) ** 2) / (4.0 * (sigma ** 2))) * np.exp(1j * k0 * x)
        return cls(psi, dx=dx, **kwargs)

    @classmethod
    def plane_wave_superposition(
        cls,
        N: int = 256,
        amplitudes: list[float] = (0.6, 0.8),
        wavenumbers: list[float] = (1.0, 3.0),
        phases: list[float] = (0.0, 0.0),
        dx: float = 0.1,
        **kwargs,
    ) -> "Wavefunction":
        """Construct an explicit superposition of plane waves."""
        x = np.arange(N) * dx
        psi = np.zeros(N, dtype=complex)
        for a, k, phi in zip(amplitudes, wavenumbers, phases):
            psi += a * np.exp(1j * (k * x + phi))
        return cls(psi, dx=dx, **kwargs)

    @classmethod
    def random_state(
        cls,
        N: int = 256,
        seed: int = 42,
        dx: float = 0.1,
        **kwargs,
    ) -> "Wavefunction":
        """Construct a maximally chaotic (high-C_s) random wavefunction."""
        rng = np.random.default_rng(seed)
        psi = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        return cls(psi, dx=dx, **kwargs)

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def __add__(self, other: "Wavefunction") -> "Wavefunction":
        """Superpose two wavefunctions: ψ_new = normalise(ψ_a + ψ_b)."""
        if self.N != other.N:
            raise ValueError(f"Grid size mismatch: {self.N} vs {other.N}")
        
        new_wf = Wavefunction(
            self._psi + other._psi,
            dx=self.dx,
            delta_omega=self.delta_omega,
            fidelity_target=self.fidelity_target,
            phase_cost_model=self.phase_cost_model,
            phase_resolution=self.phase_resolution,
        )
        # Inherit non-default bases registered in the parent states
        for name, basis in self.bases.items():
            if name != "fourier":
                new_wf.register_basis(basis)
        return new_wf

    def __mul__(self, scalar: complex) -> "Wavefunction":
        """Scale ψ by a complex scalar (result is renormalised)."""
        new_wf = Wavefunction(
            self._psi * scalar,
            dx=self.dx,
            delta_omega=self.delta_omega,
            fidelity_target=self.fidelity_target,
            phase_cost_model=self.phase_cost_model,
            phase_resolution=self.phase_resolution,
        )
        for name, basis in self.bases.items():
            if name != "fourier":
                new_wf.register_basis(basis)
        return new_wf

    def __rmul__(self, scalar: complex) -> "Wavefunction":
        return self.__mul__(scalar)

    def inner_product(self, other: "Wavefunction") -> complex:
        """Discrete inner product ⟨self|other⟩ = Σ_x ψ*(x) φ(x) dx."""
        return complex(np.sum(self._psi.conj() * other._psi) * self.dx)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(psi: np.ndarray) -> np.ndarray:
        """Normalise ψ to unit L² norm."""
        norm: float = float(np.sqrt(np.sum(np.abs(psi) ** 2)))
        if norm == 0.0:
            raise ValueError("Cannot normalise a zero wavefunction.")
        return psi / norm

    def __repr__(self) -> str:
        # Avoid computational explosion during debug print strings
        return (f"Wavefunction(N={self.N}, dx={self.dx}, "
                f"fidelity={self.fidelity_target}, model='{self.phase_cost_model}', "
                f"bases_avail={list(self.bases.keys())})")
    
