import random
import numpy as np
import sys
import Sparse_Sketching
sys.path.append("..")
import config

np.random.seed(1)
random.seed(1)

def EXP3S_1(D_i, n, _EXP3S):
    i = 0
    S = []
    R = []
    N_a = np.zeros((config.M, 1))

    
    D_i = D_i.tolist()
    D_i.sort(key=(lambda x: x[0]))
    D_i = np.array(D_i, dtype=object)
    print(D_i[0][0])
    for a in config.A:
        S_aj = []
        R_aj = []
        while (i < len(D_i) and D_i[i][0] == a):
            S_aj.append(D_i[i][1])
            R_aj.append(D_i[i][2])
            N_a[a] += 1
            i = i + 1
        S.append(np.array(S_aj))
        R.append(np.array(R_aj).reshape((-1, 1)))
    print(i)
    for a in config.A:
        # _EXP3S.Fai[a] += np.dot(S[a].T, S[a])
        # _EXP3S.B_aj[a] += np.dot(S[a].T, R[a])

        _EXP3S.Fai[a] += Sparse_Sketching.matr_multiply(S[a].T, S[a])
        _EXP3S.B_aj[a] += Sparse_Sketching.matr_multiply(S[a].T, R[a])
        
        
        # _EXP3S.Theta[a] = np.dot(np.linalg.inv(_EXP3S.Fai[a]), _EXP3S.B_aj[a])
        _EXP3S.Theta[a] = Sparse_Sketching.matr_multiply(np.linalg.inv(_EXP3S.Fai[a]), _EXP3S.B_aj[a])

        if len(_EXP3S.Theta_capital[a]) == 0:
            _EXP3S.Theta_capital[a] = _EXP3S.Theta[a]
        else:
            _EXP3S.Theta_capital[a] = np.concatenate((_EXP3S.Theta_capital[a], _EXP3S.Theta[a]), axis=1)

def EXP3S_2(s, a, n, _EXP3S):
    t = _EXP3S.Theta_capital[a]
    # R_s = np.sum(np.dot(t.T, s))
    R_s = np.sum(Sparse_Sketching.matr_multiply(t.T, s))

    _EXP3S.P[a] = np.exp(_EXP3S.eta * R_s)

    p_sum = np.sum(_EXP3S.P)
    for a in config.A:
        _EXP3S.Q[a] = _EXP3S.P[a] / p_sum
    pai = (1 - _EXP3S.delta) * _EXP3S.Q + _EXP3S.delta / config.M

    return pai
