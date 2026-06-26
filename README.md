# Wavefunction

Quantum Mechanical complex valued wavefunction with Spectral Complexity measure.

## Install

```bash
pip install wavefunction
```

## Introduction

Spectral Complexity measure for complex-valued wavefunctions.

## Theory background (Meskanen 2026 — "The Wavefunction as Compression")

The central hypothesis is that the quantum wavefunction is the universe's data-compression codec. Internal observers — themselves composed of compressed structures — perceive their constituent degrees of freedom as wave-like because they are observing *compressed information*. The codec that produces this compression is the Fourier / spectral decomposition.

### Spectral Complexity $C_s$

A wavefunction $\psi(x)$ can always be written as a superposition of spectral modes, each characterised by two attributes:

- **frequency** $\omega$ — the rate of oscillation, unbounded above zero
- **phase** $\phi$ — the offset of the oscillation, bounded in $[0, 2\pi)$

The *spectral complexity* $C_s(\psi)$ is the total continuous information cost needed to specify the set of modes that materially compose $\psi$:

$$
C_s(\psi) = \sum_i \left[ \frac{\omega_i}{\Delta \omega} + \phi_{\text{cost}}(\phi_i) \right]
$$

#### Frequency cost (dominant term)

$\frac{\omega_i}{\Delta \omega}$ is the number of resolution steps $\Delta \omega$ needed to locate frequency $\omega_i$. It is unbounded, continuous, and grows linearly with frequency. This term *dominates* $C_s$ and is the reason the measure exponentially suppresses high-frequency (rough, chaotic) states.

The identification $\Delta \omega = \hbar \ln 2$ connects the minimum frequency resolution to Planck's constant.

#### Phase cost (subdominant, bounded)

Each phase $\phi_i \in [0, 2\pi)$ requires a finite amount of information to specify. The cost is *global* over all modes: it measures how much information is needed to distinguish the phases from one another.

With only two modes at phases 0 and $\pi$, very little is needed; with many modes at crowded, uneven phases, somewhat more is required. In practice this term is bounded by $\log_2(N_{\text{modes}})$ and is a second-order correction.

The current implementation uses a simple uniform fixed cost per non-reference mode as a tractable proxy; the reference mode (highest amplitude) is exempt because only *relative* phases are observable — a global phase shift leaves $|\psi(x)|^2$ unchanged.

#### Amplitude and the fidelity engine

Amplitude does not appear as a separate encoding cost. Instead it determines *which modes are included* in the description via a power-ranked fidelity engine: modes are added in descending power order until the accumulated power reaches a target fraction of the total. Modes below this threshold are simply absent from the description — they are not part of the codec output and contribute zero complexity cost.

This correctly handles the case where many weak modes coexist with a few dominant ones: the dominant modes determine $C_s$; the weak modes are free.

#### Solomonoff suppression and the probability profile

Under Solomonoff-like induction the prior probability of a configuration is $P(\psi) \propto 2^{-C_s(\psi)}$. Because $C_s$ is a *sum* over independent modes, the probability *factorises*:

$$
P(\psi) \propto \prod_i 2^{-\omega_i / \Delta \omega}
$$

Each mode is suppressed independently and exponentially by its frequency. The resulting probability profile is Boltzmann-like with inverse temperature $\beta = \ln(2)/\Delta \omega$:

$$
P(\text{mode } i \text{ present}) \propto \exp(-\ln(2) \cdot \omega_i / \Delta \omega)
$$

Smooth, low-frequency, compressible states dominate the measure. Boltzmann brains, random fluctuations, and chaotic configurations are exponentially suppressed — not by fine-tuning, but because they require many high-frequency modes to describe.

### The conjecture $C_s \propto S_{\text{Euclidean}}$

The central open conjecture (Meskanen 2026, §9) is that the minimum spectral complexity path through configuration space coincides with the minimum Euclidean action path of standard quantum gravity. If true, quantum gravity is Solomonoff induction over compressed descriptions of geometry, and $\hbar$ is the minimum spectral resolution of a finite informational universe.
