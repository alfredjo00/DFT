import numpy as np
import plot_settings
import scipy.linalg as linalg
from hartree_fock_linear_combination import normalize_wf

plot_settings.set_params()
a0 = 5.29177210903e-1   #Ångström


def hydrogen_wf(r):
    return np.exp(-r) / np.sqrt(np.pi)


def fn_u(n_s, r):
    return np.sqrt(4 * np.pi * n_s) * r


def hartree_potential(r):
    return 1 / r - (1 + 1 / r) * np.exp(-2 * r)


def charge_density(wv_fn):
    return np.array([f ** 2 for f in wv_fn])


def energy_x(n):
    return -(3/4) * np.cbrt(3 * n / np.pi)


def energy_c(n):
    def e_c(n_r):
        r_s = np.cbrt(3 / (4 * np.pi * n_r))
        if r_s >= 1:
            gamma, beta1, beta2 = -0.1423, 1.0529, 0.3334
            return gamma / (1 + beta1 * np.sqrt(r_s) + beta2 * r_s)
        A, B, C, D = 0.0311, -0.048, 0.0020, -0.0116
        return A * np.log(r_s) + B + C * r_s * np.log(r_s) + D * r_s
    return np.array([e_c(m) for m in n])


def potential_x(n_s, R):
    u = np.sqrt(4 * np.pi * n_s) * R
    return -np.cbrt(3 * u ** 2 / (2 * np.pi ** 2 * R ** 2))


def potential_c(n, R):
    e_c = energy_c(n)
    return 2 * n * np.gradient(e_c, abs(R[1] - R[0])) + e_c


"""
    Total energy for helium atom, with electron density = 2 * n_s

"""


def ground_state_energy_lda(R, n_s, V_H, epsilon, exchange=False, correlation=False):
    u = fn_u(n_s, R)
    if exchange and not correlation:
        integrand = u**2 * (V_H/2 + potential_x(n_s, R)/4)
    elif correlation:
        integrand = u**2 * (V_H/2 + potential_x(n_s, R) + potential_c(2 * n_s, R) -
                            (energy_x(2 * n_s) + energy_c(2 * n_s)))
    else:
        integrand = u**2 * V_H/2

    potential_integration = np.trapz(x=R, y=integrand)
    return 2 * epsilon - 2 * potential_integration


def finite_difference_poisson(n_s, R):
    N = len(n_s)
    h = abs(R[1] - R[0])

    A = np.zeros((N + 1, N + 1))
    b = np.zeros(N + 1)
    A[0, 0] = 1
    A[N, N] = 1

    u_r = np.array([np.sqrt(4 * np.pi * y) * r for y, r in zip(n_s, R)])
    for i in range(1, N):
        A[i, i - 1] = 1
        A[i, i] = -2
        A[i, i + 1] = 1
        b[i] = -u_r[i-1] ** 2 * h ** 2 / (R[i-1])
    return np.linalg.solve(A, b).reshape(-1)[1:]


def finite_difference_kohn_sham(R, V):
    N = len(R)
    h = abs(R[1] - R[0])
    D = np.zeros((N, N))

    for i in range(N):
        D[i, i - 1] = -1 / (2 * h ** 2) * (i > 0)
        D[i, i] = 1 / h ** 2 + V[i]
        if i < N - 1:
            D[i, i + 1] = -1 / (2 * h ** 2)

    w, v = linalg.eigh(a=D, subset_by_index=[0, 0])
    v = normalize_wf(R, v.reshape(-1))
    return w[0], v


def method_hartree_approx_self_cons(R, exchange, correlation):
    r_max = R[-1]
    psi = np.array([min(1/(r**2), 5) for r in R])
    psi = normalize_wf(R, psi)
    n_s = charge_density(psi)
    E = []

    for i in range(30):
        U_0 = finite_difference_poisson(n_s, R)
        V_sH = np.array([u / r + 1 / r_max for u, r in zip(U_0, R)])

        if exchange or correlation:
            V_H = 2 * V_sH
            if exchange and not correlation:
                V_eff = V_H + potential_x(n_s, R) - 2/R
            else:
                V_eff = V_H + potential_x(n_s, R) + potential_c(n_s, R) - 2/R
        else:
            V_H = V_sH
            V_eff = V_H - 2/R

        epsilon, u = finite_difference_kohn_sham(R, V_eff)
        psi = u / (np.sqrt(4 * np.pi) * R)
        psi = normalize_wf(R, psi)
        n_s = charge_density(psi)
        E_0 = ground_state_energy_lda(R, n_s, V_H, epsilon, exchange, correlation)
        E.append(E_0)
        print(E_0, epsilon)

        if i > 3 and abs(E[-1] - E[-2]) < 1e-5:
            break
    return E[-1], np.sqrt(n_s), u




