
# --------------------- #
#                       #
#    Main Single Run    #
#                       #
# --------------------- #

import numpy as np
from fgen_solver.fgen_core import fgen_core
from sklearn.gaussian_process.kernels import Matern
import matplotlib.pyplot as plt
from numpy import linalg as LA
import time
# from pprint import pprint


if __name__ == '__main__':

    # seed = np.random.randint(0, 1e5)
    seed = 54
    np.random.seed(seed)

    # --------------------------- #
    #  set simulation parameters  #
    # --------------------------- #

    m = 500  # number of samples
    n = 20000  # number of features

    not0 = 10  # number of non 0 features

    domain = np.array([0, 1])  # domains of the curves
    n_eval = 1000   # number of points to construct the true predictors and the response

    mu_x = 0  # mean of the true predictors
    sd_x = 1  # standard deviation of the x Matern covariance
    l_x = 0.25  # range parameter of x Matern covariance
    nu_x = 3.5  # smoothness of x Matern covariance

    mu_eps = 0  # mean of errors
    sd_eps = 1  # standard deviation of the eps Matern covariance
    l_eps = 0.25  # range parameter of eps Matern covariance
    nu_eps = 1.5  # smoothness of eps Matern covariance

    # --------------------- #
    #  set fgen parameters  #
    # --------------------- #

    k = 5  # number of FPC scores

    c_lam = 0.8  # lam1 = alpha * c_lam * lam1_max
    alpha = 0.8  # lam2 = (1-alpha) * c_lam * lam1_max

    debias = True  # if True a linear model is performed on the selected FPCA scores to debias the estimates
    smoothing_x = True  # if true, the x curves computed from their FPC scores, are smoothed

    sgm = not0 / n  # starting value of sigma
    sgm_increase = 5  # sigma increasing factor
    sgm_change = 1  # number of iteration that we wait before incresing sigma

    use_cg = True  # decide if you want to use conjugate gradient
    r_exact = 2000  # number of features such that we start using the exact method

    mu = 0.2   # step size parameter
    step_reduce = 0.5   # step size reducing factor

    maxiter_ssn = 40  # snn max iterations
    maxiter_ssnal = 100  # snnal max iterations
    tol_ssn = 1e-6  # snn tolerance
    tol_ssnal = 1e-6  # snnal tolerance

    print_lev = 1  # decide level of printing

    # ------------------ #
    #  create variables  #
    # ------------------ #

    # create equispaced grid where the curves are evaluated at
    grid = np.linspace(domain[0], domain[1], n_eval)

    # create design matrix A
    print('')
    print('  * generating A')
    A = np.random.normal(0, 1, (m, n))
    # # if you want to specify a covariance C different from the identity matrix
    # A = np.random.multivariate_normal(np.zeros(n), C, m)
    # A = (A - A.mean(axis=0)) / A.std(axis=0)

    print('  * creating features')
    # create the features -- and their covariance using a matern process
    cov_x = sd_x ** 2 * Matern(length_scale=l_x, nu=nu_x)(grid.reshape(-1, 1))
    x_true = np.random.multivariate_normal(mu_x * np.ones(n_eval), cov_x, not0)
    # x_true = np.random.normal(0, 10, (not0, 1)) * grid

    print('  * creating errors')
    # create the errors -- and their covariance using a matern process
    cov_eps = sd_eps ** 2 * Matern(length_scale=l_eps, nu=nu_eps)(grid.reshape(-1, 1))
    eps = np.random.multivariate_normal(mu_eps * np.ones(n_eval), cov_eps, m)
    # # if you want to use snt to determine sd_eps:
    # sd_eps = np.sqrt(np.var(np.dot(A[:, 0:not0], x_true)) / 5)

    # compute the responses: if you do not center b, then you have to change the FPC basis inside!!!!
    print('  * computing b')
    b = np.dot(A[:, 0:not0], x_true) + eps
    b = b - b.mean(axis=0)

    # find b_scores, eigvals, eigfuns
    eigvals, eigfuns = LA.eigh(np.dot(b.T, b))
    b_scores = np.dot(b, eigfuns[:, -k:])

    # find lam1_max
    lam1_max = np.max(LA.norm(np.dot(A.T, b_scores), axis=1)) / alpha

    # ------------ #
    #  some plots  #
    # ------------ #

    # # plot of the features
    # plt.plot(grid, x_true.T, lw=1)
    # plt.plot(grid, mu_x * np.ones(n_eval), 'k', lw=3)
    # plt.show()

    # # plot of some of the responses
    # plt.plot(grid, b[0:10, ].T, lw=1)
    # plt.show()

    # # plot first k eigenfunctions
    # plt.plot(grid, eigfuns[:, -k:], lw=1)
    # plt.show()

    # ----------- #
    #  fgen_core  #
    # ----------- #

    print('')
    print('  * start fgen')
    print('  * sgm =', sgm)
    out_fgen = fgen_core(A=A, b=b, k=k,
                         c_lam=c_lam, alpha=alpha, lam1_max=lam1_max,
                         grid=grid,
                         b_scores=b_scores, eigvals=eigvals, eigfuns=eigfuns,
                         x0=None, y0=None, z0=None, Aty0=None,
                         debias=debias, smoothing_x=smoothing_x,
                         sgm=sgm, sgm_increase=sgm_increase, sgm_change=sgm_change,
                         step_reduce=step_reduce, mu=mu,
                         tol_ssn=tol_ssn, tol_ssnal=tol_ssnal,
                         maxiter_ssn=maxiter_ssn, maxiter_ssnal=maxiter_ssnal,
                         use_cg=use_cg, r_exact=r_exact,
                         print_lev=print_lev)

    # ------------------------------------ #
    #  plot estimated coefficients curves  #
    # ------------------------------------ #

    # plt.plot(grid, out_fgen.x_curves[0:, :].T, lw=1)
    # plt.gca().set_prop_cycle(None)
    # plt.plot(grid, x_true[0:, :].T, '--')
    # plt.show()

   