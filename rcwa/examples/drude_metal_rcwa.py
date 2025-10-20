"""Drude-model metal film analysis using rcwa.

This script reproduces a hand-written transfer-matrix calculation using the
rcwa package for a homogeneous metallic film described by a Drude
permittivity model.  It evaluates the metal permittivity, reflectance, and the
surface electric-field enhancement for TE and TM polarizations as a function
of angular frequency.

Two damping rates (gamma) are compared in order to illustrate how loss
affects the optical response of the film.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from rcwa import Layer, LayerStack, Material, Solver, Source


# ---------------------------------------------------------------------------
# Problem setup
# ---------------------------------------------------------------------------
C0 = 299_792_458.0  # Speed of light in vacuum (m/s)
EPS_INF = 1.0
WP = 10.0  # rad/s
GAMMAS = (0.5, 0.0005)  # rad/s
D_FILM = 10e-9  # 10 nm
THETA0_DEG = 45.0
THETA0 = np.deg2rad(THETA0_DEG)
N0 = 1.0  # Incident medium (air)
NS = 1.0  # Substrate (air)

OMEGA = np.linspace(1.0, 20.0, 3000)  # rad/s
LAMBDA = 2.0 * np.pi * C0 / OMEGA  # Convert angular frequency to wavelength (m)

# The rcwa Source object expects both TE and TM amplitudes to be present in
# order to avoid division-by-zero when extracting r_TE / r_TM.  We therefore
# excite with equal TE and TM amplitudes and later read out the individual
# reflection coefficients from the solver results.
PTEM = np.array([1.0, 1.0]) / np.sqrt(2.0)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def drude_permittivity(omega: np.ndarray | float, wp: float, gamma: float, eps_inf: float = 1.0) -> np.ndarray:
    """Complex permittivity for a Drude metal."""
    omega = np.asarray(omega, dtype=np.complex128)
    return eps_inf - (wp ** 2) / (omega ** 2 + 1j * gamma * omega)


def omega_from_wavelength(wavelength: np.ndarray | float) -> np.ndarray:
    wavelength = np.asarray(wavelength, dtype=np.float64)
    return 2.0 * np.pi * C0 / wavelength


def kz_in_medium(omega: np.ndarray, n: np.ndarray, kx: np.ndarray) -> np.ndarray:
    """Longitudinal wavevector component with sign chosen for passive media."""
    k0 = omega / C0
    kz_squared = (k0 * n) ** 2 - kx ** 2
    kz = np.sqrt(kz_squared + 0j)
    # Ensure Im(kz) >= 0 so that evanescent waves decay in +z.
    kz = np.where(np.imag(kz) < 0, -kz, kz)
    return kz


def admittance_te(kz: np.ndarray, omega: np.ndarray) -> np.ndarray:
    k0 = omega / C0
    return kz / k0


def admittance_tm(n: np.ndarray, kz: np.ndarray, omega: np.ndarray) -> np.ndarray:
    k0 = omega / C0
    return (n ** 2) * k0 / kz


def field_enhancement_te(r_te: np.ndarray) -> np.ndarray:
    return np.abs(1.0 + r_te) ** 2


def field_enhancement_tm(r_tm: np.ndarray, omega: np.ndarray, eps_layer: np.ndarray, theta0: float) -> np.ndarray:
    """TM field enhancement just inside the film entrance."""
    n_layer = np.sqrt(eps_layer)
    k0 = omega / C0
    kx = k0 * N0 * np.sin(theta0)
    kz0 = kz_in_medium(omega, N0 + 0j, kx)
    kz1 = kz_in_medium(omega, n_layer, kx)

    Y0 = admittance_tm(N0 + 0j, kz0, omega)
    Y1 = admittance_tm(n_layer, kz1, omega)

    A = 0.5 * ((1.0 + r_tm) + (Y0 / Y1) * (1.0 - r_tm))
    B = 0.5 * ((1.0 + r_tm) - (Y0 / Y1) * (1.0 - r_tm))

    ex_entrance = A + B
    ez_entrance = -(kx / kz1) * A + (kx / kz1) * B
    incident_norm = 1.0 + np.abs(kx / kz0) ** 2
    return (np.abs(ex_entrance) ** 2 + np.abs(ez_entrance) ** 2) / incident_norm


def build_solver(material: Material, initial_wavelength: float) -> Solver:
    source = Source(wavelength=initial_wavelength, theta=THETA0, phi=0.0, pTEM=PTEM)
    incident = Layer(er=N0 ** 2)
    film = Layer(material=material, thickness=D_FILM)
    substrate = Layer(er=NS ** 2)
    stack = LayerStack(film, incident_layer=incident, transmission_layer=substrate)
    return Solver(stack, source, n_harmonics=1)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
results_by_gamma = {}

for gamma in GAMMAS:
    # Drude material implemented as a wavelength-dependent permittivity.
    def eps_lambda(lam: float, *, gamma_val: float = gamma) -> complex:
        omega_val = omega_from_wavelength(lam)
        return drude_permittivity(omega_val, WP, gamma_val, EPS_INF)

    metal = Material(er=eps_lambda)

    solver = build_solver(metal, initial_wavelength=LAMBDA[0])
    sim_results = solver.solve(wavelength=LAMBDA)

    r_te = np.asarray(sim_results['rTE'])
    r_tm = np.asarray(sim_results['rTM'])
    r_total = np.asarray(sim_results['RTot'])

    eps_w = drude_permittivity(OMEGA, WP, gamma, EPS_INF)
    R_te = np.abs(r_te) ** 2
    R_tm = np.abs(r_tm) ** 2

    fe_te = field_enhancement_te(r_te)
    fe_tm = field_enhancement_tm(r_tm, OMEGA, eps_w, THETA0)

    results_by_gamma[gamma] = {
        'epsilon': eps_w,
        'R_tot': r_total,
        'R_te': R_te,
        'R_tm': R_tm,
        'FE_te': fe_te,
        'FE_tm': fe_tm,
    }

    # ------------------------------------------------------------------
    # Plotting for each damping rate
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    axes[0].plot(OMEGA, np.real(eps_w), label='Re($\\varepsilon$)')
    axes[0].plot(OMEGA, np.imag(eps_w), label='Im($\\varepsilon$)')
    axes[0].set_ylabel('Permittivity')
    axes[0].set_title(f'Drude metal properties (γ = {gamma} s$^{{-1}}$)')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(OMEGA, R_te, label='TE')
    axes[1].plot(OMEGA, R_tm, label='TM')
    axes[1].set_ylabel('Reflectance R')
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(OMEGA, fe_te, label='TE field enhancement')
    axes[2].plot(OMEGA, fe_tm, label='TM field enhancement')
    axes[2].set_xlabel('Angular frequency ω (rad/s)')
    axes[2].set_ylabel('Surface field enhancement')
    axes[2].legend()
    axes[2].grid(True)

    if gamma == min(GAMMAS):
        axes[2].set_yscale('linear')
    else:
        axes[2].set_yscale('log')

    fig.tight_layout()

plt.show()

# Print a concise summary that may be useful for sanity checks.
for gamma, data in results_by_gamma.items():
    eps_sample = data['epsilon'][0]
    print(f"γ = {gamma} s^-1 -> ε(ω_min) = {eps_sample:.4f}, R_TE(ω_min) = {data['R_te'][0]:.4f}, "
          f"R_TM(ω_min) = {data['R_tm'][0]:.4f}")
