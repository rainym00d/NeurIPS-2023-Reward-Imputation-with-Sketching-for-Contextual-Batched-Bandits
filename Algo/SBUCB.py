import random
import numpy as np
import datetime
import sys
sys.path.append("..")
import config
import Sparse_Sketching
import reward

np.random.seed(1)
random.seed(1)


def UCB(D_i, _UCB, REWARD):
    _d = config.d
    if REWARD == 'kernel':
        _d = config.D_kernel    

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
            if REWARD == 'kernel':
                s = reward.kernel_s(s)
                s = s.reshape((1, -1)).tolist()[0]
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

   
    for a in config.A:
        print("S shape: ", S[a].shape)
        # _UCB.Fai[a] += np.dot(S[a].T, S[a])
        # _UCB.B_aj[a] += np.dot(S[a].T, R[a])
        
        _UCB.Fai[a] += Sparse_Sketching.matr_multiply(S[a].T, S[a])
        _UCB.B_aj[a] += Sparse_Sketching.matr_multiply(S[a].T, R[a])

        

        _UCB.Theta[a] = Sparse_Sketching.matr_multiply(np.linalg.inv(_UCB.Fai[a]), _UCB.B_aj[a])
        # _UCB.Theta[a] = np.dot(np.linalg.inv(_UCB.Fai[a]), _UCB.B_aj[a])
    
