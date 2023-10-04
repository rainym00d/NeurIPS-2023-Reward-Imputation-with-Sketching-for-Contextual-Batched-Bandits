'''
BEXP3.S1-IPW(Inverse Propensity Weighting)
'''
import random
import numpy as np
from numpy.linalg.linalg import LinAlgError
import config
import Sparse_Sketching

np.random.seed(1)
random.seed(1)

def BEXP3S_1_IPW(D_i, n, _EXP3S):
    i = 0
    S = []
    R = []
    N_a = np.zeros((config.M, 1))

    D_i = D_i.tolist()
    D_i.sort(key=(lambda x: x[0]))
    D_i = np.array(D_i, dtype=object)

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

    for a in config.A:
        # _EXP3S.Fai[a] += np.dot(S[a].T, S[a])
        # _EXP3S.B_aj[a] += np.dot(S[a].T, R[a])
        _EXP3S.Fai[a] += Sparse_Sketching.matr_multiply(S[a].T, S[a])
        _EXP3S.B_aj[a] += Sparse_Sketching.matr_multiply(S[a].T, R[a])
        try:
            # _EXP3S.Theta[a] = np.dot(np.linalg.inv(_EXP3S.Fai[a]), _EXP3S.B_aj[a])
            _EXP3S.Theta[a] = Sparse_Sketching.matr_multiply(np.linalg.inv(_EXP3S.Fai[a]), _EXP3S.B_aj[a])
        except LinAlgError:
            # _EXP3S.Theta[a] = np.dot(np.linalg.pinv(_EXP3S.Fai[a]), _EXP3S.B_aj[a])
            _EXP3S.Theta[a] = Sparse_Sketching.matr_multiply(np.linalg.pinv(_EXP3S.Fai[a]), _EXP3S.B_aj[a])

        if len(_EXP3S.Theta_capital[a]) == 0:
            _EXP3S.Theta_capital[a] = _EXP3S.Theta[a]
        else:
            _EXP3S.Theta_capital[a] = np.concatenate((_EXP3S.Theta_capital[a], _EXP3S.Theta[a]), axis=1)

def BEXP3S_2_IPW(s, a, n, _EXP3S):
    '''
    policy updating
    '''
    t = _EXP3S.Theta_capital[a]
    # R_s = np.sum(np.dot(t.T, s))
    R_s = np.sum(Sparse_Sketching.matr_multiply(t.T, s))
    one_in_pai = min(100, 1 / _EXP3S.pai[a]) # different from EXP3S1
    _EXP3S.P[a] = np.exp(_EXP3S.eta * R_s * one_in_pai) #different from EXP3S1

    p_sum = np.sum(_EXP3S.P)
    for a in config.A:
        _EXP3S.Q[a] = _EXP3S.P[a] / p_sum
    _EXP3S.pai = (1 - _EXP3S.delta) * _EXP3S.Q + _EXP3S.delta / config.M

    return _EXP3S.pai
