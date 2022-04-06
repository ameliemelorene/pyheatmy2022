from numpy import float32, full, gradient, zeros
from numba import njit

from .solver import solver, tri_product

LAMBDA_W = 0.6071
RHO_W = 1000
C_W = 4185

PARAM_LIST = (
    "moinslog10K",
    "n",
    "lambda_s",
    "rhos_cs",
)


@njit()
def compute_next_temp(
    moinslog10K, n, lambda_s, rhos_cs, dts, dz, temps_init, H_res, t0s, tns, alpha=0.7
):
    N = temps_init.size
    temps = zeros((dts.size + 1, N), dtype=float32)
    temps[0] = temps_init.astype(float32)

    rho_mc_m = n * RHO_W * C_W + (1 - n) * rhos_cs
    K = 10.0 ** -moinslog10K
    lambda_m = (n * (LAMBDA_W) ** 0.5 + (1.0 - n) * (lambda_s) ** 0.5) ** 2

    ke = lambda_m / rho_mc_m
    ae = RHO_W * C_W * K / rho_mc_m

    for ix in range(1, dts.size + 1):
        dt = dts[ix - 1]

        dH_prev = zeros(N, dtype=float32)
        dH_prev[1:-1] = (H_res[ix - 1, 2:] - H_res[ix - 1, :-2]) / (2.0 * dz)
        dH_prev[0] = (H_res[ix - 1, 2] - H_res[ix - 1, 1]) / dz
        dH_prev[-1] = (H_res[ix - 1, -2] - H_res[ix - 1, -3]) / dz

        a = (-ke / dz ** 2 + dH_prev[1:] * (ae / (2 * dz))) * (1 - alpha)
        a[0] = -(1 - alpha) * (2 * ke / dz ** 2 - ae * dH_prev[0] / (2 * dz))
        a[-2] = -(1 - alpha) * (ke / dz ** 2 + ae * dH_prev[-1] / (2 * dz))
        b = full(N, (1 - alpha) * 2 * ke / dz ** 2 - 1 / dt, dtype=float32)
        b[1] = -(1 - alpha) * (-2 * ke / dz ** 2) * (3 / 2) - 1 / dt
        b[-2] = -(1 - alpha) * (-2 * ke / dz ** 2) * (3 / 2) - 1 / dt
        c = (-ke / dz ** 2 - dH_prev[:-1] * ae / (2 * dz)) * (1 - alpha)
        c[1] = -(1 - alpha) * (ke / dz ** 2 + ae * dH_prev[0] / (2 * dz))
        c[-1] = -(1 - alpha) * (2 * ke / dz ** 2 - ae * dH_prev[-1] / (2 * dz))

        lim = tri_product(a, b, c, temps[ix - 1])
        lim[0], lim[-1] = t0s[ix], tns[ix]

        dH = zeros(N, dtype=float32)
        dH[1:-1] = (H_res[ix, 2:] - H_res[ix, :-2]) / (2.0 * dz)
        dH[0] = (H_res[ix, 2] - H_res[ix, 1]) / dz
        dH[-1] = (H_res[ix, -2] - H_res[ix, -3]) / dz

        a = (ke / dz ** 2 - dH[1:] * ae / (2 * dz)) * alpha
        a[0] = alpha * (2 * ke / dz ** 2 - ae * dH[0] / (2 * dz))
        a[-2] = alpha * (ke / dz ** 2 + ae * dH[-1] / (2 * dz))
        a[-1] = 0.0
        b = full(N, -alpha * 2 * ke / dz ** 2 - 1 / dt, dtype=float32)
        b[0], b[-1] = 1.0, 1.0
        b[1] = alpha * (-2 * ke / dz ** 2) * (3 / 2) - 1 / dt
        b[-2] = alpha * (-2 * ke / dz ** 2) * (3 / 2) - 1 / dt
        c = (ke / dz ** 2 + dH[:-1] * ae / (2 * dz)) * alpha
        c[0] = 0.0
        c[1] = alpha * (ke / dz ** 2 + ae * dH[0] / (2 * dz))
        c[-1] = alpha * (2 * ke / dz ** 2 - ae * dH[-1] / (2 * dz))

        temps[ix] = solver(a, b, c, lim)
    return temps


@njit
def compute_next_h(K, Ss, dts, dz, H_init, dHs, alpha=0.7):
    dHs = dHs.astype(float32)
    dts = dts.astype(float32)

    N = H_init.size
    H_res = zeros((dts.size + 1, N), dtype=float32)
    H_res[0] = H_init.astype(float32)

    for ix in range(1, dts.size + 1):
        dt = dts[ix - 1]

        k = K * (1 - alpha) / dz ** 2

        a = full(N - 1, -k, dtype=float32)
        a[0] = -(1 - alpha) * K * 8 / (3 * (dz) ** 2)
        a[-2] = -(1 - alpha) * K * 4 / (3 * (dz) ** 2)
        b = full(N, 2 * k - Ss / dt, dtype=float32)
        b[1] = (1 - alpha) * K * 4 / (dz) ** 2 - Ss / dt
        b[-2] = (1 - alpha) * K * 4 / (dz) ** 2 - Ss / dt
        c = full(N - 1, -k, dtype=float32)
        c[1] = -(1 - alpha) * K * 4 / (3 * (dz) ** 2)
        c[-1] = -(1 - alpha) * K * 8 / (3 * (dz) ** 2)

        lim = tri_product(a, b, c, H_res[ix - 1])

        lim[0] = dHs[ix]
        lim[N - 1] = 0

        k = K * alpha / dz ** 2

        a = full(N - 1, k, dtype=float32)
        a[0] = 8 * alpha * K / (3 * ((dz) ** 2))
        a[-2] = 4 * (alpha) * K / (3 * (dz) ** 2)
        a[-1] = 0
        b = full(N, -2 * k - Ss / dt, dtype=float32)
        b[0], b[-1] = 1, 1
        b[1] = -alpha * K * 4 / ((dz) ** 2) - Ss / dt
        b[-2] = -alpha * K * 4 / ((dz) ** 2) - Ss / dt
        c = full(N - 1, k, dtype=float32)
        c[0] = 0
        c[1] = 4 * alpha * K / (3 * ((dz) ** 2))
        c[-1] = 8 * (alpha) * K / (3 * (dz) ** 2)

        H_res[ix] = solver(a, b, c, lim)
    return H_res
