import numpy as np
from wavefunction import Wavefunction

if __name__ == "__main__":
    print(__doc__)
    print("=" * 65)
    print("Demo")
    print("=" * 65)

    # 1. Gaussian packet — few dominant modes, low C_s
    gp = Wavefunction.gaussian_packet(N=256, x0=12.8, sigma=2.0, k0=2.0, dx=0.1)
    print("\n[1] Gaussian wave packet")
    gp.spectral_complexity(verbose=True)
    print(gp)

    # 2. Two-component superposition — moderate C_s
    sp = Wavefunction.plane_wave_superposition(
        N=256, amplitudes=[0.6, 0.8],
        wavenumbers=[1.0, 5.0], phases=[0.0, np.pi / 4], dx=0.1)
    print("\n[2] Two-component superposition")
    sp.spectral_complexity(verbose=True)
    print(sp)

    # 3. Random state — all modes active, high C_s
    rnd = Wavefunction.random_state(N=256, seed=0, dx=0.1)
    print("\n[3] Random (high-entropy) state")
    print(f"  C_s = {rnd.spectral_complexity():.2f}  (verbose suppressed)")
    print(rnd)

    # 4. Solomonoff weights — exponential ordering
    print("\nSolomonoff weights  2^{{-C_s}}:")
    for label, wf in [("Gaussian", gp), ("Superposition", sp), ("Random", rnd)]:
        print(f"  {label:>16s}:  {wf.solomonoff_weight():.4e}")

    # 5. Interference economy
    gp2 = Wavefunction.gaussian_packet(N=256, x0=20.0, sigma=2.0, k0=2.0, dx=0.1)
    combined = gp + gp2
    print(f"\n[5] Interference economy:")
    print(f"  C_s(ψ₁)       = {gp.spectral_complexity():.2f}")
    print(f"  C_s(ψ₂)       = {gp2.spectral_complexity():.2f}")
    print(f"  C_s(ψ₁ + ψ₂)  = {combined.spectral_complexity():.2f}")
    print("  Superposition is cheaper than two independent descriptions.")

    # 6. Fidelity engine: weak modes don't inflate C_s
    x = np.arange(64)
    psi_clean = sum(np.exp(1j * 2*np.pi*k*x/64) for k in [2, 7, 13])
    rng = np.random.default_rng(0)
    psi_noisy = psi_clean + 0.01 * sum(
        np.exp(1j * (2*np.pi*k*x/64 + rng.uniform(0, 2*np.pi)))
        for k in range(20, 50))
    wf_clean = Wavefunction(psi_clean, dx=0.1)
    wf_noisy = Wavefunction(psi_noisy, dx=0.1)
    print(f"\n[6] Fidelity engine — weak modes ignored:")
    print(f"  C_s(clean, 3 modes)          = {wf_clean.spectral_complexity():.3f}")
    print(f"  C_s(+ 30 weak noise modes)   = {wf_noisy.spectral_complexity():.3f}")
    print(f"  Modes retained (clean/noisy) = "
          f"{len(wf_clean.retained_modes())} / {len(wf_noisy.retained_modes())}")
