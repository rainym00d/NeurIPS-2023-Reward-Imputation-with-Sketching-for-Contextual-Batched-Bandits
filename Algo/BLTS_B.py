import random
import numpy as np
import datetime
import sys
sys.path.append("..")
import config
import Sparse_Sketching


np.random.seed(1)
random.seed(1)


def BLTS_B(D_i, _BLTS_B):
    
    _d = config.d

    S = []
    R = []
    N_a = np.zeros((config.M, 1))

    D_i = D_i.tolist()
    D_i.sort(key=(lambda x: x[0]))
    D_i = np.array(D_i, dtype=object)

    i = 0
    for a in config.A:
        S_aj = []
        R_aj = []
        while(i < len(D_i) and D_i[i][0] == a):
            s = D_i[i][1]
            S_aj.append(s)
            R_aj.append(D_i[i][2])
            N_a[a] += 1
            i = i+1


        if len(S_aj) != 0:
            S.append(np.array(S_aj))
            R.append(np.array(R_aj).reshape((-1, 1)))
        else:
            S.append(np.zeros((0, _d)))
            R.append(np.zeros((0, 1)))

    Monto_Carlo_times = 100 # times for Sampling
    # IPW
    Theta_sim = []
    for a in config.A:
        fai_inv = np.linalg.inv(_BLTS_B.Fai[a])
        tmp = []
        for j in range(Monto_Carlo_times):
            tmp.append(np.random.multivariate_normal(_BLTS_B.Theta_line[a].reshape(-1), (_BLTS_B.miu**2) * fai_inv, 1))
            Theta_sim.append(tmp)    

    W = []
    
    for a in config.A:
        P_a = []
        for s in S[a]:
            p_s = 0
            tmp = []
            for j in range(Monto_Carlo_times):
                for _a in config.A:
                    # tmp.append(np.dot(s, Theta_sim[_a][j].T))
                    tmp.append(Sparse_Sketching.matr_multiply(s.reshape(-1, 1), Theta_sim[_a][j].T))
                    
                p = np.argmax(tmp)
                if p == a:
                    p_s += 1
            p_s /= Monto_Carlo_times
            p_s = max(_BLTS_B.gamma, p_s)
            P_a.append(np.sqrt(1/p_s))

        W.append(np.diag(P_a))

    for a in config.A:
        # S[a] = np.dot(W[a], S[a])
        # R[a] = np.dot(W[a], R[a])
        
        S[a] = Sparse_Sketching.matr_multiply(W[a], S[a])
        R[a] = Sparse_Sketching.matr_multiply(W[a], R[a])

    for a in config.A:
        # _BLTS_B.B_aj[a] += np.dot(S[a].T, R[a])
        _BLTS_B.Fai[a] += Sparse_Sketching.matr_multiply(S[a].T, S[a])
        _BLTS_B.B_aj[a] += Sparse_Sketching.matr_multiply(S[a].T, R[a])
        fai_inv = np.linalg.inv(_BLTS_B.Fai[a])
        # _BLTS_B.Theta_line[a] = np.dot(fai_inv, _BLTS_B.B_aj[a])
        _BLTS_B.Theta_line[a] = Sparse_Sketching.matr_multiply(fai_inv, _BLTS_B.B_aj[a])

        _BLTS_B.Theta[a] = np.random.multivariate_normal(_BLTS_B.Theta_line[a].reshape(-1), (_BLTS_B.miu**2) * fai_inv, 1).reshape((-1, 1))