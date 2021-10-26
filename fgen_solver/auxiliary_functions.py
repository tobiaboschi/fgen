

# -------------------------------------------------------------- #
#                                                                #
#    auxiliary function for ssnal elastic net and SNALL lasso    #
#                                                                #
# -------------------------------------------------------------- #


import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt


def prox(v, par1, par2):

    """
    computes proximal operator for functional elastic net penalization

    """

    return v / (1 + par2) * np.repeat(np.maximum(0, 1 - par1 / LA.norm(v, axis=1)), v.shape[1]).reshape(v.shape[0], v.shape[1])
    # return v / (1 + par2) * np.repeat(np.maximum(0, 1 - par1 / LA.norm(v, axis=1)), p).reshape(v.shape[0], p)


def prox_star(v, par1, par2, t):

    """
    computes proximal operator for conjugate functional elastic net penalization

    :param v: the argument is already divided by sigma: prox_(p*/sgm)(x/sgm = v)

    """

    return v - prox(v * t, par1, par2) / t


def p_star(v, par1, par2):

    """
    computes the conjugate function of the functional elastic net penalization

    """

    if par2 > 0:
        return np.sum(np.maximum(0, LA.norm(v, axis=1) - par1) ** 2) / (2 * par2)
        # return np.sum(np.maximum(0, LA.norm(v, axis=1) - par1) ** 2) / (2 * par2)

    else:
        return np.sum(np.maximum(0, LA.norm(v, axis=1) - par1) ** 2)


def prim_obj(A, x, b, lam1, lam2):

    """
    computes primal object of the functional elastic net

    """

    return 0.5 * LA.norm(np.dot(A, x) - b) ** 2 + lam2 / 2 * LA.norm(x) ** 2 + lam1 * np.sum(LA.norm(x, axis=1))


def dual_obj(y, z, b, lam1, lam2):

    """
    computes dual object of the functional elastic net

    """

    return - (0.5 * LA.norm(y) ** 2 + np.sum(b * y) + p_star(z, lam1, lam2))


def phi_y(y, x, b, Aty, sgm, lam1, lam2):

    """
    computes phi(y) for the functional elastic net problem

    """

    return (LA.norm(y) ** 2 / 2 + np.sum(b * y) + (1 + sgm * lam2) / (2 * sgm) *
            LA.norm(prox(x - sgm * Aty, sgm * lam1, sgm * lam2)) ** 2 - LA.norm(x) ** 2 / (2 * sgm))


def grad_phi(A, y, x, b, Aty, sgm, lam1, lam2):

    """
    computes the gradient of phi(y) for the functional elastic net problem

    """

    return y + b - np.dot(A, prox(x - sgm * Aty, sgm * lam1, sgm * lam2))


def plot_cv_fgen(r, ebic, gcv, cv, alpha, grid):

    """
    plots of: r, ebic, gcv, cv for different values of alpha

    :param r: list of r_lm. Each element of the list is the r_lm values for the respective alpha in alpha_list
    :param ebic: list of ebic. Each element of the list is the ebic values for the respective alpha in alpha_list
    :param gcv: list of gcv. Each element of the list is the gcv values for the respective alpha in alpha_list
    :param cv: list of cv. Each element of the list is the cv values for the respective alpha in alpha_list
    :param alpha: vec of different value of alpha considered
    :param grid: array of all the lambda1_ratio considered (same for all alphas)

    """

    # if the inputs are not list, we create them:
    if type(r) != list:
        r_list, ebic_list, gcv_list, cv_list = list(), list(), list(), list()
        r_list.append(r)
        ebic_list.append(ebic)
        gcv_list.append(gcv)
        cv_list.append(cv)
        alpha_vec = np.array([alpha])
        n_alpha = 1
    else:
        r_list, ebic_list, gcv_list, cv_list, alpha_vec = r, ebic, gcv, cv, alpha
        n_alpha = alpha.shape[0]

    # chech if we need to print cv
    if np.sum(cv_list[0]) == - cv_list[0].shape[0]:
        cv = False
    else:
        cv = True

    fig, ax = plt.subplots(2, 2)

    # ebic
    for i in range(n_alpha):
        indx = ebic_list[i] != -1
        t = grid[:ebic_list[i].shape[0]][indx]
        ax[0, 0].plot(t, ebic_list[i][indx], label=('alpha = %.2f' % alpha_vec[i]))
    ax[0, 0].legend(loc='best')
    ax[0, 0].set_title('ebic')
    ax[0, 0].set_xlim([grid[0], grid[-1]])

    # gcv
    for i in range(n_alpha):
        indx = gcv_list[i] != -1
        t = grid[:gcv_list[i].shape[0]][indx]
        ax[0, 1].plot(t, gcv_list[i][indx], label=('alpha = %.2f' % alpha_vec[i]))
    ax[0, 1].legend(loc='best')
    ax[0, 1].set_title('gcv')
    ax[0, 1].set_xlim([grid[0], grid[-1]])

    # r_lm
    for i in range(n_alpha):
        indx = r_list[i] != -1
        t = grid[:r_list[i].shape[0]][indx]
        ax[1, 0].plot(t, r_list[i][indx], label=('alpha = %.2f' % alpha_vec[i]))
    ax[1, 0].legend(loc='best')
    ax[1, 0].set_title('selected features')
    ax[1, 0].set_xlim([grid[0], grid[-1]])

    # cv
    if cv:
        for i in range(n_alpha):
            indx = cv_list[i] != -1
            t = grid[:cv_list[i].shape[0]][indx]
            ax[1, 1].plot(t, cv_list[i][indx], label=('alpha = %.2f' % alpha_vec[i]))
        ax[1, 1].legend(loc='best')
        ax[1, 1].set_title('cross validation')
        ax[1, 1].set_xlim([grid[0], grid[-1]])

    plt.show()

