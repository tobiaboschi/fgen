

# -------------------------------------- #
#                                        #
#    Functional SsNAL ELASTIC NET PATH   #
#                                        #
# -------------------------------------- #


import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy import linalg as LA
from fgen_solver.fgen_core import fgen_core
from sklearn.model_selection import KFold
from fgen_solver.auxiliary_functions import plot_cv_fgen
import skfda



class OutputPath:

    """
    Definition of the output class

    """

    def __init__(self, best_ebic_model, r, indx, ebic, gcv, cv, c_lam, alpha, lam1, lam2, lam1_max, times_array,
                 time_path, time_cv, time_total, b_scores, eigvals, eigfuns):
        self.best_ebic_model = best_ebic_model
        self.r = r
        self.indx = indx
        self.ebic = ebic
        self.gcv = gcv
        self.cv = cv
        self.c_lam = c_lam
        self.alpha = alpha
        self.lam1 = lam1
        self.lam2 = lam2
        self.lam1_max = lam1_max
        self.times_array = times_array
        self.time_path = time_path
        self.time_cv = time_cv
        self.time_total = time_total
        self.b_scores = b_scores
        self.eigvals = eigvals
        self.eigfuns = eigfuns


def fgen_path(A, b, k,
              c_lam_vec, alpha, lam1_max=None,
              grid=None,
              b_scores=None, eigvals=None, eigfuns=None,
              x0=None, y0=None, z0=None, Aty0=None,
              debias=True, smoothing_x=True,
              cv=False, n_folds=10, max_selected=None,
              sgm=5e-3, sgm_increase=5, sgm_change=3,
              step_reduce=0.5, mu=0.2,
              tol_ssn=1e-6, tol_ssnal=1e-6,
              maxiter_ssn=50, maxiter_ssnal=100,
              use_cg=False, r_exact=2e4,
              plot=False, print_lev=1):

    """
    --------------------------------------------------------------------------
    ssnal algorithm to solve the elastic net for a list of lambda1 and lambda2
    --------------------------------------------------------------------------

    INPUT PARAMETERS:
    --------------------------------------------------------------------------------------------------------------------
    :param A: design matrix (m x n)
    :param b: response vector (m x n_eval). Each row contains the evaluations of a response function.
    :param k: number of FPCs of b to consider
    :param c_lam_vec: np.array to determine the path of lambas
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
    :param smoothing_x: True/False. If true, the curves of the best ebic model are smoothed
    :param max_selected: if given, the algorithm stops when a number of features > max_selected is selected
    :param cv: True/False. I true, a cross validation is performed
    :param n_folds: number of folds to perform the cross validation
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
    :param plot: True/False. If true a plot of r, gcv, extended bic and cv (if cv == True) is displayed
    :param print_lev: different level of printing (0, 1, 2, 3, 4)
    --------------------------------------------------------------------------------------------------------------------

    OUTPUT: OutputPath object with the following attributes
    --------------------------------------------------------------------------------------------------------------------
    :return best_ebic_model: an OutputCore object of the model with the best ebic
    :return r: np.array, number of selected features for each value of c_lam
    :return indx: list, position of selected features for each value of c_lam
    :return ebic: np.array, extended bic for each value of c_lam
    :return gcv: np.array, gcv for each value of c_lam
    :return cv: np.array, cv for each value of c_lam
    :return c_lam: same as input
    :return alpha: same as input
    :return lam1_max: same as input
    :return lam1: np.array, lasso penalization for each value of c_lam
    :return lam2: np.array, ridge penalization for each value of c_lam
    :return times_array: array, time to compute the solution for each value of c_lam
    :return time_path: time to compute the solution path
    :return time_cv: time to perform cross validation
    :return time_path: total time
    :return b_scores: same as input
    :return eigvals: same as input
    :return eigfuns: same as input
    --------------------------------------------------------------------------------------------------------------------

    """

    # -------------------------- #
    #    initialize variables    #
    # -------------------------- #

    m, n = A.shape
    n_eval = b.shape[1]

    # b = b - b.mean(axis=0)

    if x0 is None:
        x0 = np.zeros((n, k))
    if y0 is None:
        y0 = np.zeros((m, k))
    if z0 is None:
        z0 = np.zeros((n, k))
    if grid is None:
        grid = np.linspace(0, 1, n_eval)

    sgm0 = sgm

    if max_selected is None:
        max_selected = n

    if c_lam_vec is None:
        c_lam_vec = np.geomspace(1, 0.01, num=100)

    n_lam1 = c_lam_vec.shape[0]
    n_lam1_stop = n_lam1

    convergence = True
    reached_max = False
    best_ebic_flag = False

    # -------------------------- #
    #    create output arrays    #
    # -------------------------- #

    ebic_vec, gcv_vec = - np.ones([n_lam1]), - np.ones([n_lam1])
    r_vec, indx_list = - np.ones([n_lam1]), list()
    times_vec, iters_vec = - np.ones([n_lam1]), - np.ones([n_lam1])

    # ----------------------- #
    #    perform FPCA of b    #
    # ----------------------- #

    # print('')
    if b_scores is None or eigfuns is None or eigvals is None:
        eigvals, eigfuns = LA.eigh(np.dot(b.T, b))
        b_scores = np.dot(b, eigfuns[:, -k:])

    norm_captured = (np.cumsum(np.flip(eigvals)) / np.sum(eigvals))[k]

    print('')
    if print_lev > 1:
        print('-----------------------------------------------------------------------')
        print(' norm approximation = %.4f' % norm_captured)
        print('-----------------------------------------------------------------------')

    # -----------------------#
    #    compute lam1 max    #
    # ---------------------- #

    if lam1_max is None:
        # lam1_max2 = np.max(LA.norm(np.dot(A.T, b), axis=1))
        lam1_max = np.max(LA.norm(np.dot(A.T, b_scores), axis=1)) / alpha

    lam1_vec = alpha * c_lam_vec * lam1_max
    lam2_vec = (1 - alpha) * c_lam_vec * lam1_max

    # ---------------------- #
    #    solve full model    #
    # ---------------------- #

    if print_lev > 0:
        print('-----------------------------------------------------------------------')
        print(' * solving full model * ')
        print('-----------------------------------------------------------------------')

    start_path = time.time()

    for i in range(n_lam1):

        if print_lev > 2:
            print('-----------------------------------------------------------------------')
            print(' FULL MODEL:  c_lam = %.2f  |  sigma0 = %.2e' % (c_lam_vec[i], sgm0))
            print('-----------------------------------------------------------------------')

        # ------------------- #
        #    perform ssnal    #
        # ------------------- #

        fit = fgen_core(A=A, b=b, k=k,
                                  c_lam=c_lam_vec[i], alpha=alpha, lam1_max=lam1_max,
                                  grid=grid,
                                  b_scores=b_scores, eigvals=eigvals, eigfuns=eigfuns,
                                  x0=x0, y0=y0, z0=z0, Aty0=Aty0,
                                  debias=debias, smoothing_x=False,
                                  sgm=sgm0, sgm_increase=sgm_increase, sgm_change=sgm_change,
                                  step_reduce=step_reduce, mu=mu,
                                  tol_ssn=tol_ssn, tol_ssnal=tol_ssnal,
                                  maxiter_ssn=maxiter_ssn, maxiter_ssnal=maxiter_ssnal,
                                  use_cg=use_cg, r_exact=r_exact,
                                  print_lev=print_lev - 3)

        # ----------------------- #
        #    check convergence    #
        # ----------------------- #

        if not fit.convergence:
            convergence = False
            break

        # ---------------------------- #
        #    update starting values    #
        # ---------------------------- #

        x0, y0, z0, Aty0, sgm0 = fit.x_scores, fit.y, fit.z, fit.Aty, fit.sgm

        # -------------------------- #
        #    update output arrays    #
        # -------------------------- #

        times_vec[i], r_vec[i], iters_vec[i] = fit.time, fit.r, fit.iters
        indx_list.append(fit.indx)

        if r_vec[i] > 0:
            ebic_vec[i], gcv_vec[i] = fit.ebic, fit.gcv

            # ---------------------------- #
            #    update best_ebic_model    #
            # ---------------------------- #

            if best_ebic_flag:
                if fit.ebic < best_ebic_model.ebic:
                    best_ebic_model = fit
            else:
                best_ebic_model = fit
                best_ebic_flag = True

        # --------------------------------------- #
        #    check number of selected features    #
        # --------------------------------------- #

        if r_vec[i] > max_selected:
            n_lam1_stop = i + 1
            reached_max = True
            break

    # ------------------- #
    #    end full model   #
    # ------------------- #

    time_path = time.time() - start_path

    if not convergence:
        print('-----------------------------------------------------------------------')
        print(' FSsNAL-EN has not converged for c_lam = %.2f' % c_lam_vec[i])
        print('-----------------------------------------------------------------------')

    if reached_max and print_lev > 0:
        print('-----------------------------------------------------------------------')
        print(' max number of features has been selected')
        print('-----------------------------------------------------------------------')

    # -------------- #
    #    start cv    #
    # -------------- #

    time_cv = 0
    cv_mat = - np.ones([n_lam1_stop, n_folds])

    if cv and convergence:

        print('-----------------------------------------------------------------------')
        print(' * performing cv *  ')
        print('-----------------------------------------------------------------------')

        x0_cv, z0_cv = np.zeros((n, k)), np.zeros((n, k))
        Aty0_cv = None
        sgm_cv = sgm
        fold = 0

        start_cv = time.time()

        # ------------- #
        #    split A    #
        # ------------- #

        kf = KFold(n_splits=n_folds)
        kf.get_n_splits(A)

        # -------------------- #
        #    loop for folds    #
        # -------------------- #

        for train_index, test_index in kf.split(A):

            A_train, A_test = A[train_index], A[test_index]
            b_scores_train, b_scores_test = b_scores[train_index], b_scores[test_index]

            y0_cv = np.zeros((np.shape(train_index)[0], k))

            # ------------------------ #
            #    loop for lam_ratio    #
            # ------------------------ #

            for i_cv in tqdm(range(n_lam1_stop)):

                # ------------------- #
                #    perform ssnal    #
                # ------------------- #

                fit_cv = fgen_core(A=A_train, b=b, k=k,
                                   c_lam=c_lam_vec[i_cv], alpha=alpha, lam1_max=lam1_max,
                                   grid=grid,
                                   b_scores=b_scores_train, eigvals=eigvals, eigfuns=eigfuns,
                                   x0=x0_cv, y0=y0_cv, z0=z0_cv, Aty0=Aty0_cv,
                                   debias=debias, smoothing_x=False,
                                   sgm=sgm_cv, sgm_increase=sgm_increase, sgm_change=sgm_change,
                                   step_reduce=step_reduce, mu=mu,
                                   tol_ssn=tol_ssn, tol_ssnal=tol_ssnal,
                                   maxiter_ssn=maxiter_ssn, maxiter_ssnal=maxiter_ssnal,
                                   use_cg=use_cg, r_exact=r_exact,
                                   print_lev=0)

                # ------------------- #
                #    update cv mat    #
                # ------------------- #

                cv_mat[i_cv, fold] = LA.norm(np.dot(A_test, fit_cv.x_scores) - b_scores_test) ** 2

                # ---------------------------- #
                #    update starting values    #
                # ---------------------------- #

                if i_cv == n_lam1_stop:
                    x0_cv, y0_cv, z0_cv, Aty0_cv, sgm_cv = None, None, None, None, sgm

                else:
                    x0_cv, y0_cv, z0_Cv, Aty0_cv, sgm_cv = fit_cv.x_scores, fit_cv.y, fit_cv.z, fit_cv.Aty, fit_cv.sgm

            # ------------------------ #
            #    end loop for lam1    #
            # ------------------------ #

            fold += 1

        # ------------ #
        #    end cv    #
        # ------------ #

        time_cv = time.time() - start_cv

    # ----------------------------------------- #
    #   smoothing best ebic curves if needed    #
    # ----------------------------------------- #

    if smoothing_x:
        fd_object = skfda.FDataGrid(best_ebic_model.x_curves, grid)
        n_basis = int(np.minimum(n_eval / 2, np.maximum(np.sqrt(n_eval), 30)))
        best_ebic_model.x_curves = fd_object.to_basis(skfda.representation.basis.BSpline(n_basis=n_basis))
        best_ebic_model.x_curves = best_ebic_model.x_curves.evaluate(grid)[:, :, 0]

    # ------------------------- #
    #    print final results    #
    # ------------------------- #

    if cv:
        cv_vec = cv_mat.mean(1) / m
    else:
        cv_vec = - np.ones([n_lam1_stop])

    time_total = time_path + time_cv
    time.sleep(0.1)

    if print_lev > 0:
        print('-----------------------------------------------------------------------')
        print(' total time:  %.4f' % time_total)
        if cv:
            print('-----------------------------------------------------------------------')
            print('  time path:  %.4f' % time_path)
            print('-----------------------------------------------------------------------')
            print('    time cv:  %.4f' % time_cv)
        print('-----------------------------------------------------------------------')

    if print_lev > 1:

        # --------------------------- #
        #    printing final matrix    #
        # --------------------------- #

        print()
        print_matrix1 = np.stack((c_lam_vec[:n_lam1_stop], r_vec[:n_lam1_stop],
                                  ebic_vec[:n_lam1_stop], gcv_vec[:n_lam1_stop], cv_vec), -1)  #
        df1 = pd.DataFrame(print_matrix1, columns=['c_lam', 'r', 'ebic', 'gcv', 'cv'])
        pd.set_option('display.max_rows', df1.shape[0] + 1)
        print(df1.round(2))

        print('-----------------------------------------------------------------------')

        r_not0 = r_vec > 0
        argmin_ebic = np.argmin(ebic_vec[r_not0]) + 1
        argmin_gcv = np.argmin(gcv_vec[r_not0]) + 1
        print_matrix2 = np.array([[argmin_ebic, c_lam_vec[argmin_ebic], r_vec[argmin_ebic]],
                                  [argmin_gcv,   c_lam_vec[argmin_gcv],  r_vec[argmin_gcv]],
                                  [-1,                                   -1,                 -1]])
        if cv:
            argmin_cv = np.argmin(cv_vec[r_not0[:n_lam1_stop]]) + 1
            print_matrix2[2, :] = [argmin_cv, c_lam_vec[argmin_cv], r_vec[argmin_cv]]

        df2 = pd.DataFrame(print_matrix2, index=['argmin_ebic', 'argmin_gcv', 'argmin_cv'], columns=['pos', 'c_lam', 'r'])
        pd.set_option('display.max_rows', df2.shape[0] + 1)
        print(df2.round(2))

    # ---------------------- #
    #    plot if required    #
    # ---------------------- #

    if plot:
        plot_cv_fgen(r_vec, ebic_vec, gcv_vec, cv_vec, alpha, c_lam_vec)

    # ------------------- #
    #    create output    #
    # ------------------- #

    out = OutputPath(best_ebic_model, r_vec[:n_lam1_stop], indx_list, ebic_vec[:n_lam1_stop], gcv_vec[:n_lam1_stop], cv_vec,
                     c_lam_vec[:n_lam1_stop], alpha, lam1_vec[:n_lam1_stop], lam2_vec[:n_lam1_stop], lam1_max,
                     times_vec[:n_lam1_stop], time_path, time_cv, time_total, b_scores, eigvals, eigfuns)

    return out