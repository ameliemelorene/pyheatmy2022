from numpy import float32, full, zeros
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
def compute_T(
    moinslog10K, n, lambda_s, rhos_cs, all_dt, dz, H_res, H_riv, H_aq, T_init, T_riv, T_aq, alpha=0.3
):
    """
    Computes T(z, t) by solving the heat equation : dT/dt = ke Delta T + ae nabla H nabla T
    In matrix form, we have : A*T_{t+1} = B*T_t + c.
    Arguments :
        - moinslog10K = - log10(K), where K = permeability
        - n = porosity
        - lambda_s = thermal conductivity
        - rho_cs = density
        - times = list of times at which we want to compute T.
        - dz = spatial discretization step
        - H_res = array of H(z, t)
        - H_riv = list of H in the river for each time
        - H_aq = list of H in the aquifer for each time
        - T_init = list of T(z, t=0)
        - T_riv = list of T in the river for each time
        - T_aq = list of T in the aquifer for each time
        - alpha = parameter of the semi-implicit scheme
    """
    rho_mc_m = n * RHO_W * C_W + (1 - n) * rhos_cs
    K = 10.0 ** -moinslog10K
    lambda_m = (n * (LAMBDA_W) ** 0.5 + (1.0 - n) * (lambda_s) ** 0.5) ** 2

    ke = lambda_m / rho_mc_m
    ae = RHO_W * C_W * K / rho_mc_m

    n_cell = len(T_init)
    n_times = len(all_dt) + 1

    # First we need to compute the gradient of H(z, t)

    nablaH = zeros((n_cell, n_times), float32)

    nablaH[0, :] = 2*(H_res[0, :] - H_riv)/dz

    for i in range(1, n_cell - 1):
        nablaH[i, :] = (H_res[i+1, :] - H_res[i-1, :])/(2*dz)

    nablaH[n_cell - 1, :] = 2*(H_aq - H_res[n_cell - 1, :])/dz

    # Now we can compute T(z, t)

    T_res = zeros((n_cell, n_times), float32)
    T_res[:, 0] = T_init

    for j, dt in enumerate(all_dt):
        # Compute T at time times[j+1]

        # Defining the 3 diagonals of B
        lower_diagonal = (ke*alpha/dz ** 2) - (alpha*ae/(2*dz)) * nablaH[1:, j]
        lower_diagonal[-1] = 4*ke*alpha/(3*dz**2)

        diagonal = full(n_cell, 1/dt - 2*ke*alpha/dz**2, float32)
        diagonal[0] = 1/dt - 4*ke*alpha/dz**2 + 2*alpha*ae*nablaH[0, j]/dz
        diagonal[-1] = 1/dt - 4*ke*alpha/dz**2 - \
            2*alpha*ae*nablaH[n_cell-1, j]/dz

        upper_diagonal = (ke*alpha/dz ** 2) + (alpha*ae/(2*dz)) * nablaH[1:, j]
        upper_diagonal[0] = 4*ke*alpha/(3*dz**2)

        # Defining c
        c = zeros(n_cell, float32)
        c[0] = (8*ke*(1-alpha) / (3*dz**2) - 2*(1-alpha)*ae*nablaH[0, j]/dz) * \
            T_riv[j+1] + (8*ke*alpha / (3*dz**2) - 2*alpha *
                          ae*nablaH[0, j]/dz) * T_riv[j]
        c[-1] = (8*ke*(1-alpha) / (3*dz**2) + 2*(1-alpha)*ae*nablaH[n_cell - 1, j]/dz) * \
            T_aq[j+1] + (8*ke*alpha / (3*dz**2) + 2*alpha *
                         ae*nablaH[n_cell - 1, j]/dz) * T_aq[j]

        B_fois_T_plus_c = tri_product(
            lower_diagonal, diagonal, upper_diagonal, T_res[:, j]) + c

        # Defining the 3 diagonals of A
        lower_diagonal = - (ke*(1-alpha)/dz ** 2) + \
            ((1-alpha)*ae/(2*dz)) * nablaH[1:, j]
        lower_diagonal[-1] = - 4*ke*(1-alpha)/(3*dz**2)

        diagonal = full(n_cell, 1/dt + 2*ke*(1-alpha)/dz**2, float32)
        diagonal[0] = 1/dt + 4*ke*(1-alpha)/dz**2 - \
            2*(1-alpha)*ae*nablaH[0, j]/dz
        diagonal[-1] = 1/dt + 4*ke*(1-alpha)/dz**2 + \
            2*(1-alpha)*ae*nablaH[n_cell-1, j]/dz

        upper_diagonal = - (ke*(1-alpha)/dz ** 2) - \
            ((1-alpha)*ae/(2*dz)) * nablaH[1:, j]
        upper_diagonal[0] = - 4*ke*(1-alpha)/(3*dz**2)

        T_res[:, j+1] = solver(lower_diagonal, diagonal,
                               upper_diagonal, B_fois_T_plus_c)

    return T_res


@njit
def compute_H(K, Ss, all_dt, isdtconstant, dz, H_init, H_riv, H_aq, alpha=0.3):
    """
    Computes H(z, t) by solving the diffusion equation : Ss dH/dt = K Delta H
    In matrix form, we have : A*H_{t+1} = B*H_t + c.
    Arguments :
        - K = permeability
        - Ss = specific emmagasinement
        - times = list of times at which we want to compute H.
        - dz = spatial discretization step
        - H_init = list of H(z, t=0)
        - H_riv = list of H in the river for each time
        - H_aq = list of H in the aquifer for each time
        - alpha = parameter of the semi-implicit scheme
    """
    n_cell = len(H_init)
    n_times = len(all_dt) + 1

    H_res = zeros((n_cell, n_times), float32)
    H_res[:, 0] = H_init

    KsurSs = K/Ss

    # Check if dt is constant :
    if isdtconstant:  # dt is constant so A and B are constant
        dt = all_dt[0]

        # Defining the 3 diagonals of B
        lower_diagonal_B = full(n_cell - 1, KsurSs*alpha/dz**2, float32)
        lower_diagonal_B[-1] = 4*KsurSs*alpha/(3*dz**2)

        diagonal_B = full(n_cell, 1/dt - 2*KsurSs*alpha/dz**2, float32)
        diagonal_B[0] = 1/dt - 4*KsurSs*alpha/dz**2
        diagonal_B[-1] = 1/dt - 4*KsurSs*alpha/dz**2

        upper_diagonal_B = full(n_cell - 1, KsurSs*alpha/dz**2, float32)
        upper_diagonal_B[0] = 4*KsurSs*alpha/(3*dz**2)

        # Defining the 3 diagonals of A
        lower_diagonal_A = full(
            n_cell - 1, - KsurSs*(1-alpha)/dz**2, float32)
        lower_diagonal_A[-1] = - 4*KsurSs*(1-alpha)/(3*dz**2)

        diagonal_A = full(n_cell, 1/dt + 2*KsurSs*(1-alpha)/dz**2, float32)
        diagonal_A[0] = 1/dt + 4*KsurSs*(1-alpha)/dz**2
        diagonal_A[-1] = 1/dt + 4*KsurSs*(1-alpha)/dz**2

        upper_diagonal_A = full(
            n_cell - 1, - KsurSs*(1-alpha)/dz**2, float32)
        upper_diagonal_A[0] = - 4*KsurSs*(1-alpha)/(3*dz**2)

        for j in range(n_times - 1):
            # Compute H at time times[j+1]

            # Defining c
            c = zeros(n_cell, float32)
            c[0] = (8*KsurSs / (2*dz**2)) * \
                (alpha*H_riv[j+1] + (1-alpha)*H_riv[j])
            c[-1] = (8*KsurSs / (2*dz**2)) * \
                (alpha*H_aq[j+1] + (1-alpha)*H_aq[j])

            B_fois_H_plus_c = tri_product(
                lower_diagonal_B, diagonal_B, upper_diagonal_B, H_res[:, j]) + c

            H_res[:, j+1] = solver(lower_diagonal_A, diagonal_A,
                                   upper_diagonal_A, B_fois_H_plus_c)
    else:  # dt is not constant so A and B and not constant
        for j, dt in enumerate(all_dt):
            # Compute H at time times[j+1]

            # Defining the 3 diagonals of B
            lower_diagonal = full(n_cell - 1, KsurSs*alpha/dz**2, float32)
            lower_diagonal[-1] = 4*KsurSs*alpha/(3*dz**2)

            diagonal = full(n_cell, 1/dt - 2*KsurSs*alpha/dz**2, float32)
            diagonal[0] = 1/dt - 4*KsurSs*alpha/dz**2
            diagonal[-1] = 1/dt - 4*KsurSs*alpha/dz**2

            upper_diagonal = full(n_cell - 1, KsurSs*alpha/dz**2, float32)
            upper_diagonal[0] = 4*KsurSs*alpha/(3*dz**2)

            # Defining c
            c = zeros(n_cell, float32)
            c[0] = (8*KsurSs / (2*dz**2)) * \
                (alpha*H_riv[j+1] + (1-alpha)*H_riv[j])
            c[-1] = (8*KsurSs / (2*dz**2)) * \
                (alpha*H_aq[j+1] + (1-alpha)*H_aq[j])

            B_fois_H_plus_c = tri_product(
                lower_diagonal, diagonal, upper_diagonal, H_res[:, j]) + c

            # Defining the 3 diagonals of A
            lower_diagonal = full(
                n_cell - 1, - KsurSs*(1-alpha)/dz**2, float32)
            lower_diagonal[-1] = - 4*KsurSs*(1-alpha)/(3*dz**2)

            diagonal = full(n_cell, 1/dt + 2*KsurSs*(1-alpha)/dz**2, float32)
            diagonal[0] = 1/dt + 4*KsurSs*(1-alpha)/dz**2
            diagonal[-1] = 1/dt + 4*KsurSs*(1-alpha)/dz**2

            upper_diagonal = full(
                n_cell - 1, - KsurSs*(1-alpha)/dz**2, float32)
            upper_diagonal[0] = - 4*KsurSs*(1-alpha)/(3*dz**2)

            H_res[:, j+1] = solver(lower_diagonal, diagonal,
                                   upper_diagonal, B_fois_H_plus_c)

    return H_res
