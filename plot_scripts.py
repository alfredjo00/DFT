from hartree_fock_linear_combination import iterative_method, wave_function, normalize_wf
from finite_diff_methods import hydrogen_wf, charge_density, finite_difference_kohn_sham, finite_difference_poisson
from finite_diff_methods import hartree_potential, method_hartree_approx_self_cons, ground_state_energy_lda
import matplotlib.pyplot as plt
import plot_settings
import numpy as np
import os

plot_settings.set_params(text_size=27)

a0 = 5.29177210903e-1  # Ångström


def plot_hartree_lin_comb():
    alpha = [0.297104, 1.236745, 5.749982, 38.216677]
    E, C = iterative_method(alpha)

    R = np.linspace(0, 5, 200)
    y = np.array([wave_function(r, alpha, C) for r in R])
    y = normalize_wf(R * a0, y)

    rho = [4 * np.pi * r ** 2 * f ** 2 for r, f in zip(R * a0, y)]
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(R * a0, y, '-k', label=rf'$\phi(r), \, E_0 = {E[-1]:.5f}\, E_h$')
    ax.plot(R * a0, rho, '--b', label=rf'$\rho(r) = 4 \pi r^2 \psi(r), \, E_0 = {E[-1]:.5f}\, E_h$')
    ax.set_ylabel("Radial probability density")
    ax.set_xlabel(r"Radius $r$ (Å)")
    ax.legend()

    fig.tight_layout()
    fig.savefig("./graphs/hartree_linear_comb.jpg")


def plot_finite_diff_poisson():
    R2 = 20
    n = 200
    R = np.linspace(R2 / n, R2, n)
    wave_fn = hydrogen_wf(R)
    n_s = charge_density(wave_fn)
    y = finite_difference_poisson(n_s, R)

    V = [(U / r + 1 / R2) for r, U in zip(R, y)]
    V_hartree = [hartree_potential(r) for r in R]

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.set_xlabel(r"Radius $r$ (Å)")
    ax.set_ylabel(r"Energy ($E_h$)")
    ax.plot(R * a0, y, '--b', label=r'$U_0(r)$')
    ax.plot(R * a0, V, '-r', label=r'$V(r) = U_0(r)/r + 1/r_{max}$')
    ax.plot(R * a0, V_hartree, ':k', label=r'$V_H(r)$')
    ax.legend()
    fig.tight_layout()
    fig.savefig("./graphs/fin_diff_poisson.jpg")


def plot_finite_diff_kohn():
    n, r_max = 500, 10
    R = np.linspace(r_max / n, r_max, n)
    V = -1 / R
    epsilon, y, _ = finite_difference_kohn_sham(R, V)

    psi = y / (np.sqrt(4 * np.pi) * R)
    psi = normalize_wf(R, psi)

    fig, ax = plt.subplots(figsize=(8, 5))

    psi_H = hydrogen_wf(R)
    ax.set_xlabel(r"Radius $r$ (Å)")
    ax.set_ylabel("Probability amplitude density")

    ax.plot(R * a0, psi, '-b', label=fr'$\psi(r), \,\, E_0={epsilon:.5f}\,E_h$')

    ax.plot(R * a0, psi_H, ':k', label=r'Hydrogen $\psi(r) =  \frac{e^{-r}}{\sqrt{\pi}}, \,\, E_0 = 0.5\, E_h$')

    ax.legend()
    fig.tight_layout()
    fig.savefig("./graphs/fin_diff_kohn.jpg", dpi=100)


def plot_grid_convergence():
    E = []
    r_max = np.linspace(5, 75, 20)
    grid_density_0 = 5/a0

    for r in r_max:
        R = np.arange(0, r, 1 / grid_density_0)[1:]
        eps, _, _ = method_hartree_approx_self_cons(R=R, exchange=False, correlation=False)
        E.append(eps)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].plot(r_max * a0, E, '--ok', mfc="None")
    ax[0].set_title(rf"Max radius convergence, grid density = $10\,/\,$Å")
    ax[0].set_xlabel(r"Max radius $r_{max}$ (Å)")
    ax[0].set_ylabel(r"Ground state energy $E_0\,\, (E_h)$")

    E = []
    r_max = 30/a0

    grid_density = np.linspace(10, 75, 20)
    grid_N = []
    for g in grid_density:
        print(g)
        R = np.arange(0, r_max, 1 / g)[1:]
        eps, _, _ = method_hartree_approx_self_cons(R=R, exchange=False, correlation=False)
        E.append(eps)
        grid_N.append(len(R))

    ax[1].plot(grid_N, E, '--ok', mfc="None")
    ax[1].set_title(rf"Grid convergence, $r_{{max}}={r_max*a0}\,$Å")
    ax[1].set_xlabel(r"Grid points $N$")
    ax[1].set_ylabel(r"Ground state energy $E_0\,\, (E_h)$")
    fig.tight_layout()

    fig.savefig("./graphs/convergence.jpg", dpi=100)


def plot_self_cons_hartree(exchange=True, correlation=True):
    name_exchange = "_exch"
    name_correlation = "_corr"
    fig_name = ""

    if exchange:
        fig_name += name_exchange
    if correlation:
        fig_name += name_correlation

    r_max = 100
    grid_density = 80

    R = np.arange(0, r_max, 1 / grid_density)[1:]
    N = len(R)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.set_xlabel("Radius $r$ (Å)")
    ax.set_ylabel("Radial electron probability density")

    eps, psi, _ = method_hartree_approx_self_cons(R=R, exchange=exchange, correlation=correlation)
    psi = normalize_wf(R * a0, psi)

    ax.plot(R[:N // 10] * a0, psi[:N // 10], '-k', label=rf'$\psi(r)\,\,, E_0 = {eps:.5f}\, E_h.$ ')
    ax.plot(R[:N // 10] * a0, 4 * np.pi * psi[:N // 10] ** 2 * (R[:N // 10] * a0) ** 2, '--b',
            label=rf'$\rho(r) = 4 \pi r^2 |\psi(r)|^2, \, E_0 = {eps:.5f}\, E_h.$ ')
    ax.legend()
    fig.tight_layout()

    fig.savefig(f"./graphs/hartree_self_cons{fig_name}.jpg")


def plot_self_cons_hartree_all():
    r_max = 30/a0
    R = np.linspace(0, r_max, 1001)[1:]
    N = len(R)

    fig, ax = plt.subplots(1,2, figsize=(13, 5))

    ax[0].set_xlabel("Radius $r$ (Å)")
    ax[1 ].set_xlabel("Radius $r$ (Å)")
    exchangs = [False, True, True]
    correlations = [False, False, True]
    titles = ["", "Exchange: ", "Exch. + Cor.: "]
    styles = ["--k", "-.b", ":r"]
    for ex, cor, t, s in zip(exchangs, correlations, titles, styles):
        eps, psi, u = method_hartree_approx_self_cons(R=R, exchange=ex, correlation=cor)

        psi = normalize_wf(R * a0, psi)
        ax[0].plot(R[:N // 35] * a0, psi[:N // 35], s, label=t + rf'$\psi(r)\,\,, E_0 = {eps:.5f}\, E_h.$ ')
        ax[1].plot(R[:N // 35] * a0, u[:N // 35], s, label=t + rf'$u(r)$ ')

    ax[0].set_ylabel("Probability density")
    ax[1].set_ylabel("Potential ($E_h$)")
    ax[0].legend()
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(f"./graphs/hartree_self_cons_all.jpg", dpi=100)
