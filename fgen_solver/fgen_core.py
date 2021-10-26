

# --------------------------------------- #
#                                         #
#    Functional SSsNAL ELASTIC NET CORE   #
#                                         #
# --------------------------------------- #


import time
import numpy as np
from numpy import linalg as LA
import fgen_solver.auxiliary_functions as AF
from scipy.linalg import block_diag
import scipy.sparse.linalg as ss_LA
import skfda


class OutputCore:

    """
    Definition of the output class

    """

    def __init__(self, x_curves, x_scores, b_scores, y, z, r, indx, ebic, gcv, sgm, c_lam, alpha, lam1_max,
                 lam1, lam2, time_tot, iters, Aty, eigvals, eigfuns, grid, convergence):
        self.x_curves = x_curves
        self.x_scores = x_scores
        self.b_scores = b_scores
        self.y = y
        self.z = z
        self.r = r
        self.indx = indx
        self.ebic = ebic
        self.gcv = gcv
        self.sgm = sgm
        self.c_lam = c_lam
        self.alpha = alpha
        self.lam1_max = lam1_max
        self.lam1 = lam1
        self.lam2 = lam2
        self.time = time_tot
        self.iters = iters
        self.Aty = Aty
        self.eigvals = eigvals
        self.eigfuns = eigfuns
        self.grid = grid
        self.convergence = convergence



def fgen_core(A, b, k,
              c_lam=None, alpha=None, lam1_max=None,
              grid=None,
              b_scores=None, eigvals=None, eigfuns=None,
              x0=None, y0=None, z0=None, Aty0=None,
              debias=True, smoothing_x=True,
              sgm=5e-3, sgm_increase=5, sgm_change=3,
              step_reduce=0.5, mu=0.2,
              tol_ssn=1e-6, tol_ssnal=1e-6,
              maxiter_ssn=50, maxiter_ssnal=100,
              use_cg=False, r_exact=2e4,
              print_lev=1):

    """
    --------------------------------------------------------------------------------
    ssnal algorithm to solve the elastic net for fixed values of lambda1 and lambda2
    --------------------------------------------------------------------------------

    INPUT PARAMETERS:
    --------------------------------------------------------------------------------------------------------------------
    :param A: design matrix (m x n)
    :param b: response vector (m x n_eval). Each row contains the evaluations of a response function. 
    :param k: number of FPCs of b to consider
    :param c_lam: to determine lam1 and lam2, ratio of lam_max considered
    :param alpha: we have lam1 = alpha * c_lam * lam1_max, lam2 = (1 - alpha) * c_lam * lam1_max
    :param lam1_max: smallest values of lam1 that selects 0 features. If it is None, it is computed inside the function
    :param grid: the grid of the points where b has been evaluated. Needed to set the proper domain of the curves
    :param b_scores: FPCA scores of the response. If it is None, it is computed inside the function
    :param eigvals: eigenvalues of the FPCA for b. If it is None, it is computed inside the function
    :param eigfuns: eigenfunctions of the FPCA for b. If it is None, it is computed inside the function
    :param x0: initial value for the variable of the primal problem (n x p) -- vector 0 if not given
    :param y0: initial value fot the first variable of the dual problem  (m x p) -- vector of 0 if not given
    :param z0: initial value for the second variable of the dual problem (n x p) -- vector of 0 if not given
    :param Aty0: np.dot(A.transpose(), y0) (n x p)
    :param debias: True/False. If true a linear regression is performed on the FPC selected scores
    :param smoothing_x: True/False. If true, the x curves computed from their FPC scores, are smoothed
    :param sgm: starting value of the augmented lagrangian parameter sigma
    :param sgm_increase: increasing factor of sigma
    :param sgm_change: we increase sgm -- sgm *= sgm_increase -- every sgm_change iterations
    :param step_reduce: dividing factor of the step size during the linesearch
    :param mu: multiplicative factor fot the lhs of the linesearch condition
    :param tol_ssn: tolerance for the ssn algorithm
    :param tol_ssnal: global tolerance of the ssnal algorithm
    :param maxiter_ssn: maximum number of iterations for ssn
    :param maxiter_ssnal: maximum number of global iterations
    :param use_cg: True/False. If true, the conjugate gradient method is used to find the direction of the ssn
    :param r_exact: number of features such that we start using the exact method
    :param print_lev: different level of printing (0, 1, 2)
    --------------------------------------------------------------------------------------------------------------------

    OUTPUT: OutputCore object with the following attributes
    --------------------------------------------------------------------------------------------------------------------
    :return x_curves: curves computed from the estimated FPC scores of x
    :return x_scores: estimated FPCA scores of the primal variable
    :return b_scores: FPCA scores of the response
    :return y: optimal value of the first dual variable
    :return z: optimal value of the second dual variable
    :return r: number of selected features
    :return indx: position of the selected features
    :return ebic: extended bic (if debias is True, it is computed on the debiased estimates)
    :return gcv: gcv (if debias is True, it is computed on the debiased estimates)
    :return sgm: final value of the augmented lagrangian parameter sigma
    :return c_lam: same as input
    :return alpha: same as input
    :return lam1_max: same as input
    :return lam1: lasso penalization
    :return lam2: ridge penalization
    :return time: total time of ssnal
    :return iters: total ssnal's iteration
    :return Aty: np.dot(A.T(), y) computed at the optimal y. Useful to implement warmstart
    :return eigvals: same as input
    :return eigfuns: same as input
    :return grid: same as input
    :return convergence: True/False. If false the algorithm has not converged
    --------------------------------------------------------------------------------------------------------------------

    """

    # -------------------------- #
    #    initialize variables    #
    # -------------------------- #

    m, n = A.shape
    n_eval = b.shape[1]

    # b = b - b.mean(axis=0)

    x = x0
    y = y0
    z = z0
    Aty = Aty0

    if x is None:
        x = np.zeros((n, k))
    if y is None:
        y = np.zeros((m, k))
    if z is None:
        z = np.zeros((n, k))
    if Aty0 is None:
        Aty = np.dot(A.T, y)
    if grid is None:
        grid = np.linspace(0, 1, n_eval)

    # integration constant
    # I = np.sqrt(domain[1] - domain[0]) / n_eval

    convergence_ssnal = False
    # norm_captured = None

    # ----------------------- #
    #    perform FPCA of b    #
    # ----------------------- #

    # print('')
    if b_scores is None or eigfuns is None or eigvals is None:
        eigvals, eigfuns = LA.eigh(np.dot(b.T, b))
        b_scores = np.dot(b, eigfuns[:, -k:])

    norm_captured = (np.cumsum(np.flip(eigvals)) / np.sum(eigvals))[k]

    if print_lev > 1:
        print('')
        print('  -------------------------------------------------------------------')
        print('  norm approximation = %.4f' % norm_captured)
        print('  -------------------------------------------------------------------')

    # ------------------------------------- #
    #    compute lam1 max, lam1 and lam2    #
    # ------------------------------------- #

    if lam1_max is None:
        # lam1_max2 = np.max(LA.norm(np.dot(A.T, b), axis=1))
        lam1_max = np.max(LA.norm(np.dot(A.T, b_scores), axis=1)) / alpha

    lam1 = alpha * c_lam * lam1_max
    lam2 = (1 - alpha) * c_lam * lam1_max

    if print_lev > 1:
        print('  lam1_max = %.3f  |  lam1 = %.3f   |   lam2 = %.4f   ' % (lam1_max, lam1, lam2))
        print('  -------------------------------------------------------------------')

    # ---------------------------- #
    #    start ssnal iterations    #
    # ---------------------------- #

    start_ssnal = time.time()

    for it_ssnal in range(maxiter_ssnal):

        if print_lev > 1:
            print('')
            print('  ssnal iteration = %.f  |  sgm = %.2e' % (it_ssnal + 1, sgm))
            print('  -------------------------------------------------------------------')

        # --------------- #
        #    start ssn    #
        # --------------- #

        start_ssn = time.time()

        convergence_ssn = False
        x_tilde = x - sgm * Aty

        for it_ssn in range(maxiter_ssn):

            # ---------------------------- #
            #    select active features    #
            # ---------------------------- #

            norms_vec = LA.norm(x_tilde, axis=1)
            indx = (norms_vec > sgm * lam1).reshape(n)
            x_tildeJ = x_tilde[indx, :]
            xJ = x[indx, :]
            AJ = A[:, indx]
            AJty = np.dot(AJ.T, y)
            normsJ = norms_vec[indx]

            m, r = AJ.shape

            # ------------------------- #
            #    compute direction d    #
            # ------------------------- #

            if r == 0:
                method = 'E '
                # gradient when we do not select any columns
                rhs = - (y + b_scores)
                d = rhs

            else:

                rhs = - AF.grad_phi(A, y, x, b_scores, Aty, sgm, lam1, lam2).reshape(m * k)

                # ------------------------ #
                #    compute delta prox    #
                # ------------------------ #

                delta_prox = (1 / (1 + sgm * lam2) *
                              ((1 - sgm * lam1 / normsJ).reshape(r, 1, 1) * np.eye(k) +
                               (sgm * lam1 / normsJ ** 3).reshape(r, 1, 1) * np.einsum('...i,...j', x_tildeJ, x_tildeJ)))

                AJ_kron = np.kron(AJ, np.eye(k))

                # ------------------- #
                #    standard case    #
                # ------------------- #

                if m <= r:

                    # compute hessian
                    H = np.eye(m * k) + sgm * np.dot(np.dot(AJ_kron, block_diag(*delta_prox)), AJ_kron.T)

                    # conjugate method
                    if r * k > r_exact and use_cg:
                        method = 'CG'
                        d = (ss_LA.cg(H, rhs, tol=1e-04, maxiter=1000)[0]).reshape(m, k)

                    # exact method:
                    else:
                        method = 'E '
                        d = LA.solve(H, rhs).reshape(m, k)

                # ---------------------- #
                #    Woodbury formula    #
                # ---------------------- #

                else:

                    # find P inverse
                    invP = block_diag(*LA.inv(delta_prox))
                    # invP = block_diag(*LA.solve(delta_prox, np.eye(p) * np.ones((n, 1, 1))))

                    # conjugate method
                    if r * k > r_exact and use_cg:
                        method = 'CG'
                        d_temp = ss_LA.cg(invP / sgm + np.dot(AJ_kron.T, AJ_kron), np.dot(AJ_kron.T, rhs), tol=1e-04, maxiter=1000)[0]
                        d = (rhs - np.dot(AJ_kron, d_temp)).reshape(m, k)

                    # exact method:
                    else:
                        method = 'E '
                        d_temp = LA.solve(invP / sgm + np.dot(AJ_kron.T, AJ_kron), np.dot(AJ_kron.T, rhs))
                        d = (rhs - np.dot(AJ_kron, d_temp)).reshape(m, k)

            # ------------------------------ #
            #    linesearch for step size    #
            # ------------------------------ #

            step_size = 1

            rhs_term_1 = AF.phi_y(y, xJ, b_scores, AJty, sgm, lam1, lam2)
            rhs_term_2 = np.sum(rhs.reshape(m, k) * d)
            # rhs_term_1 = phi_y(y, x, b, Aty, sgm, lam1, lam2)

            while True:
                y_new = y + step_size * d
                Aty_new = np.dot(A.T, y_new)
                if AF.phi_y(y_new, x, b_scores, Aty_new, sgm, lam1, lam2) <= rhs_term_1 + mu * step_size * rhs_term_2:
                    break
                step_size *= step_reduce

            # ---------------------- #
            #    update variables    #
            # ---------------------- #

            y = y_new
            Aty = Aty_new
            z = AF.prox_star(x / sgm - Aty, sgm * lam1, sgm * lam2, sgm)
            x_tilde = x - sgm * Aty
            x_temp = x_tilde - sgm * z

            # --------------------------- #
            #    ssn convergence check    #
            # --------------------------- #

            if r > 0:
                kkt1 = np.sum(LA.norm(np.dot(AJ, x_temp[indx, :]) - b_scores - y, axis=1)) / (1 + np.sum(LA.norm(b_scores, axis=1)))
                # kkt1 = (np.sum(LA.norm(np.dot(AJ, x_temp[indx, :]) - b_scores - y, axis=1)) /
                #         (1 + np.sum(LA.norm(b_scores, axis=1)) + np.sum(LA.norm(x_temp, axis=1))))
            else:
                kkt1 = np.sum(LA.norm(np.dot(A, x_temp) - b_scores - y, axis=1)) / (1 + np.sum(LA.norm(b_scores, axis=1)))


            if print_lev > 1:
                if it_ssn + 1 > 9:
                    space = ''
                else:
                    space = ' '
                print(space, '  %.f| ' % (it_ssn + 1),  method, ' kkt1 = %.2e  -  step_size = %.1e  -  r = %.f' % (kkt1, step_size, r), sep='')

            if kkt1 < tol_ssn or r == 0:
                convergence_ssn = True
                break

        # ------------- #
        #    end ssn    #
        # ------------- #

        time_ssn = time.time() - start_ssn

        if print_lev > 1:
            print('  -------------------------------------------------------------------')
            print('  ssn time = %.4f' % time_ssn)
            print('  -------------------------------------------------------------------')

        if not convergence_ssn:
            print('\n \n')
            print('  * ssn DOES NOT CONVERGE: try to increase the number of ssn iterations or to reduce the lambdas')
            break

        # ----------------------------- #
        #    ssnal convergence check    #
        # ----------------------------- #

        x = x_temp
        xJ = x[indx, :]
        zJ = z[indx, :]

        # compute kkt3
        kkt3 = np.sum(LA.norm(z + Aty, axis=1)) / (1 + np.sum(LA.norm(z, axis=1)) + np.sum(LA.norm(y, axis=1)))

        # compute objective functions
        prim = AF.prim_obj(AJ, xJ, b_scores, lam1, lam2)  # lam1J
        dual = AF.dual_obj(y, zJ, b_scores, lam1, lam2)  # lam1J
        dual_gap = np.abs(prim - dual)/(prim + dual)

        if print_lev > 1:
            print('  kkt3 = %.5e  -  dual gap = %.5e' % (kkt3, dual_gap))

        if kkt3 < tol_ssnal and dual_gap < 1e-3:
            convergence_ssnal = True
            it_ssnal += 1
            break

        if np.mod(it_ssnal + 1, sgm_change) == 0:
            sgm *= sgm_increase

    # --------------- #
    #    end ssnal    #
    # --------------- #

    ssnal_time = time.time() - start_ssnal

    ebic, gcv, x_curves = None, None, None
    x_scores = x

    if r > 0:

        # -------------------------------- #
        #    debias and model selection    #
        # -------------------------------- #

        if debias:
            # perform linear regression on the the selected features
            x_scores[indx, :] = LA.solve(np.dot(AJ.T, AJ), np.dot(AJ.T, b_scores))

        res = b_scores - np.dot(AJ, x_scores[indx, :])

        # compute dof
        df_core = LA.inv(np.dot(AJ.T, AJ) + lam2 * np.eye(r))
        df = np.trace(np.dot(np.dot(AJ, df_core), AJ.T))

        # compute model selection criteria
        rss = LA.norm(res) ** 2
        gcv = k * rss / (m - k * df) ** 2  # m * rss / (m - p * df) ** 2
        ebic = k * np.log(rss / (m * k)) + k * df * np.log(m * k) / m + k * df * np.log(n) / m
        # matlab ebic: mulitply everything for m: m * p * np.log(rss / (m * p)) + p * df * np.log(m * p) + p * df * np.log(n)

        # ------------------------------------------------- #
        #    compute coefficients curves from FCP scores    #
        # ------------------------------------------------- #

        x_curves = np.zeros((r, n_eval))

        for j in range(k):
            x_curves += np.outer(x_scores[indx, -j], eigfuns[:, -j])

        # --------------------------------- #
        #   applying smoothing if needed    #
        # --------------------------------- #

        if smoothing_x:
            n_basis = int(np.minimum(n_eval / 2, np.maximum(np.sqrt(n_eval), 30)))
            fd_object = skfda.FDataGrid(x_curves, grid)
            x_curves = fd_object.to_basis(skfda.representation.basis.BSpline(n_basis=n_basis))
            x_curves = x_curves.evaluate(grid)[:, :, 0]

    # ---------------------------- #
    #    printing final results    #
    # ---------------------------- #

    if convergence_ssn:

        if print_lev > 0:

            print('')
            print('  ==================================================')
            print('   * iterations ......... %.f' % it_ssnal)
            print('   * ssnal time ......... %.4f' % ssnal_time)
            print('   * prim object ........ %.4e' % prim)
            print('   * dual object ........ %.4e' % dual)
            print('   * kkt3 ............... %.4e' % kkt3)
            print('   * selected features .. %.f' % r)
            if r > 0:
                print('   * ebic ............... %.4f' % ebic)
                print('   * gcv ................ %.4f' % gcv)
            print('   * norm captured ...... %.4f' % norm_captured)
            print('  ==================================================')
            print(' ')

        if not convergence_ssnal:
            print('\n')
            print('   * FSsNAL-EN HAS NOT CONVERGED:')
            print('     (try to increase the number of iterations)')
            print('\n')

    # ------------------- #
    #    create output    #
    # ------------------- #

    out = OutputCore(x_curves, x_scores, b_scores, y, z, r, indx, ebic, gcv, sgm, c_lam, alpha, lam1_max, lam1, lam2,
                     ssnal_time, it_ssnal, Aty, eigfuns, eigvals, grid, convergence_ssnal)

    return out

