import pytest
import numpy as np
from wavefunction import SpectralMode, Wavefunction

# ---------------------------------------------------------------------------
# Tests for SpectralMode
# ---------------------------------------------------------------------------

def test_spectral_mode_initialization():
    """Ensure SpectralMode properties match specifications (e.g., phase wrapping)."""
    # Amplitude should always be non-negative
    m1 = SpectralMode(frequency=2.5, amplitude=-1.2, phase=0.0)
    assert m1.amplitude == 1.2
    
    # Phase should wrap cleanly within [0, 2pi)
    m2 = SpectralMode(frequency=-1.0, amplitude=0.5, phase=2.5 * np.pi)
    assert pytest.approx(m2.phase) == 0.5 * np.pi

    m3 = SpectralMode(frequency=0.0, amplitude=1.0, phase=-0.5 * np.pi)
    assert pytest.approx(m3.phase) == 1.5 * np.pi


# ---------------------------------------------------------------------------
# Tests for Wavefunction Construction & Attributes
# ---------------------------------------------------------------------------

def test_wavefunction_normalization():
    """Verify that Wavefunction automatically normalizes input data."""
    raw_psi = np.array([3.0 + 4j, 0.0, 0.0, 0.0])
    wf = Wavefunction(raw_psi, dx=1.0)
    
    # The L2 discrete norm should equal 1.0
    assert pytest.approx(np.sum(np.abs(wf.psi) ** 2)) == 1.0
    # Born rule array sanity check
    assert pytest.approx(np.sum(wf.probability_density)) == 1.0

def test_wavefunction_zero_norm_raises_error():
    """An all-zero array cannot be normalized and should raise an exception."""
    zero_psi = np.zeros(16)
    with pytest.raises(ValueError, match="Cannot normalise a zero wavefunction"):
        Wavefunction(zero_psi)

def test_grid_derived_properties():
    """Check coordinates, sizes, and natural default values."""
    N = 8
    dx = 0.5
    raw_psi = np.ones(N)
    wf = Wavefunction(raw_psi, dx=dx)
    
    assert wf.N == N
    assert np.allclose(wf.x, np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]))
    
    expected_delta_omega = 2.0 * np.pi / (N * dx)
    assert pytest.approx(wf.delta_omega) == expected_delta_omega
    assert pytest.approx(wf.hbar_identified) == expected_delta_omega / np.log(2.0)


# ---------------------------------------------------------------------------
# Tests for Fidelity Engine & Modes
# ---------------------------------------------------------------------------

def test_fidelity_engine_greedy_accumulation():
    """Verify that the fidelity engine slices off low power noise bins."""
    N = 128
    dx = 0.1
    delta_omega = 2.0 * np.pi / (N * dx)  # ~0.49087
    
    # Choose exact grid frequencies (e.g., bin 4 and bin 20) to eliminate leakage
    k1 = 4 * delta_omega
    k2 = 20 * delta_omega
    
    wf = Wavefunction.plane_wave_superposition(
        N=N, amplitudes=[1.0, 0.01], wavenumbers=[k1, k2], dx=dx,
        fidelity_target=0.999  # Excludes the tiny second mode
    )
    
    modes = wf.retained_modes()
    # High-power mode carries >99.9% of total variance, so weak mode is ignored
    assert len(modes) == 1 
    
    # Relax target to capture both clean bins
    wf_inclusive = Wavefunction.plane_wave_superposition(
        N=N, amplitudes=[1.0, 0.01], wavenumbers=[k1, k2], dx=dx,
        fidelity_target=1.0
    )
    assert len(wf_inclusive.retained_modes()) == 2


# ---------------------------------------------------------------------------
# Tests for Spectral Complexity Mechanics
# ---------------------------------------------------------------------------

def test_global_phase_invariance():
    """Global phase factor shifts should not affect complexity or reference flags."""
    wf_base = Wavefunction.gaussian_packet(N=64, k0=2.0)
    wf_shifted = wf_base * np.exp(1j * 1.234) # arbitrary global phase rotation
    
    assert pytest.approx(wf_base.spectral_complexity()) == wf_shifted.spectral_complexity()
    assert pytest.approx(wf_base.solomonoff_weight()) == wf_shifted.solomonoff_weight()

def test_complexity_hierarchy():
    """Chaotic configurations must present vastly larger complexity profiles."""
    N = 128
    dx = 0.1
    # Simple low-frequency pure state vs random noise
    smooth_wf = Wavefunction.plane_wave_superposition(N=N, amplitudes=[1.0], wavenumbers=[0.0], dx=dx)
    chaotic_wf = Wavefunction.random_state(N=N, dx=dx, seed=123)
    
    assert smooth_wf.spectral_complexity() < chaotic_wf.spectral_complexity()
    assert smooth_wf.solomonoff_weight() > chaotic_wf.solomonoff_weight()

def test_reference_mode_exempt_from_phase_cost():
    """Verify the reference mode yields 0 phase cost; others pay phase_resolution."""
    N = 64
    dx = 0.1
    delta_omega = 2.0 * np.pi / (N * dx)  # ~0.98175
    
    # Choose exact grid frequencies (bin 0 and bin 5) to eliminate leakage
    k1 = 0.0
    k2 = 5 * delta_omega
    
    wf = Wavefunction.plane_wave_superposition(
        N=N, amplitudes=[1.0, 1.0], wavenumbers=[k1, k2], dx=dx,
        fidelity_target=1.0, phase_resolution=1.5
    )
    
    modes = wf.retained_modes()
    # Without leakage, we get exactly the 2 discrete modes we generated
    assert len(modes) == 2
    
    # 0.0 frequency mode is our reference (lowest |w| fallback for tie-breaker)
    # total cost = (0.0 / d_omega + 0.0) + (|k2| / d_omega + 1.5)
    expected_freq_cost = abs(k2) / wf.delta_omega  # matches exactly 5.0
    expected_phase_cost = 1.5
    assert pytest.approx(wf.spectral_complexity()) == expected_freq_cost + expected_phase_cost


# ---------------------------------------------------------------------------
# Tests for Mathematical Invariants & Derived Metrics
# ---------------------------------------------------------------------------

def test_solomonoff_bounds_and_profiles():
    """Validate limits for weights and dictionary profiles mapping."""
    wf = Wavefunction.gaussian_packet(N=64)
    weight = wf.solomonoff_weight()
    
    # Probability bounds
    assert 0.0 < weight <= 1.0
    
    suppressions = wf.mode_suppression_factors()
    retained_freqs = [m.frequency for m in wf.retained_modes()]
    
    assert set(suppressions.keys()) == set(retained_freqs)
    for freq, factor in suppressions.items():
        assert 0.0 < factor <= 1.0


# ---------------------------------------------------------------------------
# Tests for Linear Algebra Operators
# ---------------------------------------------------------------------------

def test_arithmetic_and_inner_product():
    """Validate vector space rules and basic inner products."""
    N = 32
    dx = 0.2
    wf1 = Wavefunction.plane_wave_superposition(N=N, amplitudes=[1.0], wavenumbers=[1.0], dx=dx)
    wf2 = Wavefunction.plane_wave_superposition(N=N, amplitudes=[1.0], wavenumbers=[2.0], dx=dx)
    
    # Mismatched grid checks
    wf_bad = Wavefunction.plane_wave_superposition(N=N+10, dx=dx)
    with pytest.raises(ValueError, match="Grid size mismatch"):
        _ = wf1 + wf_bad

    # Test superposition scaling operation paths
    wf_sum = wf1 + wf2
    assert isinstance(wf_sum, Wavefunction)
    assert pytest.approx(np.sum(np.abs(wf_sum.psi) ** 2)) == 1.0
    
    # Check scaling mechanics
    wf_scaled = wf1 * 3.0
    assert pytest.approx(np.sum(np.abs(wf_scaled.psi) ** 2)) == 1.0
    
    wf_rscaled = 3.0 * wf1
    assert np.allclose(wf_scaled.psi, wf_rscaled.psi)

    # Unitary normalization validation checks via self-inner-product
    # <psi|psi> * dx for a normalized state will return exactly 1.0 * dx * (1/dx sum logic)
    # wait, inner_product explicitly includes * self.dx multiplication at the end
    # since sum(|psi|^2) == 1, inner_product(wf, wf) is exactly 1.0 * dx.
    assert pytest.approx(wf1.inner_product(wf1)) == 1.0 * dx


def test_repr():
    """Verify string debugging readout formats correctly without breaking."""
    wf = Wavefunction.gaussian_packet(N=32)
    rep = repr(wf)
    assert "Wavefunction" in rep
    assert "C_s=" in rep
    assert "modes=" in rep
