import random
import numpy as np
import datetime
import Sparse_Sketching
import sys
sys.path.append("../")
import config
import reward

np.random.seed(1)
random.seed(1)


def Algo_PRUCB(D_i, PR_UCB, REWARD):
    cal_reward = reward.REWARD_DICT[REWARD]
    
    _d = config.d
    if REWARD == 'kernel':
        _d = config.D_kernel

    S = []
    R = []
    S_hat = []
    R_hat = []

   
    D_i = D_i.tolist()
    D_i.sort(key=(lambda x: x[0]))
    D_i = np.array(D_i, dtype=object)

    for a in config.A:
        i = 0
        S_aj = []
        R_aj = []
        S_hat_aj = []
        R_hat_aj = []
        PR_UCB_psi_inv = np.linalg.inv(PR_UCB.psi[a])
        while (i < len(D_i)):
            s = D_i[i][1]
            # if reward fuction is kernel, must change dimension of s
            if REWARD == 'kernel':
                s = reward.kernel_s(s)
                s = s.reshape((1, -1)).tolist()[0]
            if D_i[i][0] == a:
                S_aj.append(s)
                R_aj.append(D_i[i][2])
            else:
                S_hat_aj.append(s)
                s = np.array(s).reshape((_d, 1))
                tau = D_i[i][3][a]
                c = D_i[i][4][a]

                t2 = min(tau + PR_UCB.beta*c, 1)
                R_hat_aj.append(t2)

            i = i + 1

        if len(S_aj) != 0:
            S.append(np.array(S_aj))
            R.append(np.array(R_aj).reshape((-1, 1)))
        else:
            S.append(np.zeros((0, _d)))
            R.append(np.zeros((0, 1)))

        if len(S_hat_aj) != 0:
            S_hat.append(np.array(S_hat_aj))
            R_hat.append(np.array(R_hat_aj).reshape((-1, 1)))
        else:
            S_hat.append(np.zeros((0, _d)))
            R_hat.append(np.zeros((0, 1)))


    for a in config.A:
        # PR_UCB.Fai[a] += np.dot(S[a].T, S[a])
        # PR_UCB.B_aj[a] += np.dot(S[a].T, R[a])
        PR_UCB.Fai[a] += Sparse_Sketching.matr_multiply(S[a].T, S[a])
        PR_UCB.B_aj[a] += Sparse_Sketching.matr_multiply(S[a].T, R[a])

        # PR_UCB.Fai_hat[a] = PR_UCB.eta * PR_UCB.Fai_hat[a] + np.dot(S_hat[a].T, S_hat[a])
        # PR_UCB.B_aj_hat[a] = PR_UCB.eta * PR_UCB.B_aj_hat[a] + np.dot(S_hat[a].T, R_hat[a])
        PR_UCB.Fai_hat[a] = PR_UCB.eta * PR_UCB.Fai_hat[a] + \
            Sparse_Sketching.matr_multiply(S_hat[a].T, S_hat[a])
        PR_UCB.B_aj_hat[a] = PR_UCB.eta * PR_UCB.B_aj_hat[a] + \
            Sparse_Sketching.matr_multiply(S_hat[a].T, R_hat[a])

        PR_UCB.psi[a] = np.identity(_d) + PR_UCB.Fai[a] + PR_UCB.gamma * PR_UCB.Fai_hat[a]
        t1 = np.linalg.inv(PR_UCB.psi[a])
        t2 = PR_UCB.B_aj[a] + PR_UCB.gamma * PR_UCB.B_aj_hat[a]
        # PR_UCB.Theta[a] = np.dot(t1, t2)
        PR_UCB.Theta[a] = Sparse_Sketching.matr_multiply(t1, t2)
