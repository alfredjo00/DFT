import numpy as np
import scipy.linalg as linalg


def xi_fn(r, alpha):
    return np.exp(-alpha * r ** 2)


def wave_function(r, alpha, C):
    return np.sum([c * xi_fn(r, a) for a, c in zip(alpha, C)])


def normalize_wf(x, y):
    Y = [4 * np.pi * f**2 * r**2 for f, r in zip(y, x)]
    norm = np.sqrt(np.trapz(x=x, y=Y))
    return y/norm


def matrix_h(alpha):
    def element_A(i, j):
        return -4 * np.pi / (alpha[i] + alpha[j])

    def element_T(i, j):
        return 3 * alpha[i] * alpha[j] * np.pi ** (3 / 2) / ((alpha[i] + alpha[j]) ** (5 / 2))

    mat_T = np.fromfunction(lambda i, j: element_T(i, j), (4, 4), dtype=int)
    mat_A = np.fromfunction(lambda i, j: element_A(i, j), (4, 4), dtype=int)
    return mat_T + mat_A


def matrix_S(alpha):
    mat = np.fromfunction(lambda i, j: (np.pi / (alpha[i] + alpha[j])) ** 1.5, (4, 4), dtype=int)
    return mat


def matrix_Q(alpha):
    def matrix_element_Q(indices):
        assert (len(alpha) == 4 and len(indices) == 4)
        p, q, r, s = indices
        a = 2 * np.pi ** 2.5
        b = (alpha[p] + alpha[q]) * (alpha[r] + alpha[s]) * np.sqrt(alpha[p] + alpha[q] + alpha[r] + alpha[s])
        return a / b
    mat = np.fromfunction(lambda i, j, k, l: matrix_element_Q(np.array([i, j, k, l], dtype=int)), (4, 4, 4, 4))
    return mat


def normalize_C(C, S):
    a = C @ S @ C
    return np.array([c / np.sqrt(a) for c in C])


def matrix_F(C, h, S, Q):
    C = normalize_C(C=C, S=S)

    return h + C @ Q @ C


"""
    Solves generalised eigenvalue problem:
        F @ C = E' S @ C 
    
    returns:
    eigenvector corresponding to the lowest eigenvalue
"""


def solve_eigenvalue(C, h, S, Q):
    F = matrix_F(C=C, h=h, S=S, Q=Q)

    _, v = linalg.eigh(a=F, b=S, subset_by_index=[0, 0])
    return v.reshape(-1)


"""
    Determines the state's energy.
"""


def state_energy(C, h, S, Q):
    C = normalize_C(C, S)
    a = 2 * (C @ h @ C)
    b = C @ (C @ Q @ C) @ C
    return a + b


def iterative_method(alpha):
    alpha = np.array(alpha)
    h, S, Q = matrix_h(alpha), matrix_S(alpha), matrix_Q(alpha)

    C = [1, -1, 1, -1]
    E = []
    for i in range(50):
        C = solve_eigenvalue(C=C, h=h, S=S, Q=Q)
        E.append(state_energy(C=C, h=h, S=S, Q=Q))
        print(E[-1], C)
        if i > 3 and abs(E[i] - E[i - 1]) < 1e-7:
            break
    return E, C
