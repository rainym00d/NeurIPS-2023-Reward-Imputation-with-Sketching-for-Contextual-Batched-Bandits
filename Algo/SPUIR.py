import random
import numpy as np
import pandas as pd
import Sparse_Sketching
import datetime
import sys
sys.path.append("..")
import config
import reward

np.random.seed(1)
random.seed(1)


def Algo_PRUCB(D_i, PR_UCB, REWARD):
    _d = config.d
    if REWARD == 'kernel':
        _d = config.D_kernel

    S = []
    R = []
    S_hat = []
    R_hat = []


    D_i = D_i.tolist()
    D_i.sort(key=(lambda x:x[0]))
    D_i = np.array(D_i, dtype=object)

    for a in config.A:
        i = 0
        S_aj = []
        R_aj = []
        S_hat_aj = []
        R_hat_aj = []
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
            S_hat.append(np.array(S_hat_aj,dtype=object))
            R_hat.append(np.array(R_hat_aj,dtype=object).reshape((-1,1)))
        else:
            S_hat.append(np.zeros((0, _d)))
            R_hat.append(np.zeros((0,1)))

    # sparse sketching
    S_flag_Sketch = [] # whether sketch or not
    PPII = []
    PPII_R = []
    for i in range(len(S)):
        
        if(S[i].shape[0] < PR_UCB.sketch_size):
            S_flag_Sketch.append(False)
            PPII.append(0) # padding
            PPII_R.append(0)
            continue

        S_flag_Sketch.append(True)
        
        PI = Sparse_Sketching.Sparse_Transform(c = PR_UCB.sketch_size, N_aj=S[i].shape[0], p=Sparse_Sketching.SS_p)
        PPII.append(PI.transform(S[i]))
        PPII_R.append(PI.transform(R[i]))
        
        
    PPII_hat = []
    PPII_R_hat = []
    S_hat_flag_SKetch = []
    for i in range(len(S_hat)):
        if(S_hat[i].shape[0] < PR_UCB.sketch_size):
            S_hat_flag_SKetch.append(False)
            PPII_hat.append(0) # padding
            PPII_R_hat.append(0) 
            continue
        
        S_hat_flag_SKetch.append(True)

        PI = Sparse_Sketching.Sparse_Transform(c = PR_UCB.sketch_size, N_aj=S_hat[i].shape[0], p=Sparse_Sketching.SS_p)
        PPII_hat.append(PI.transform(S_hat[i])) 
        PPII_R_hat.append(PI.transform(R_hat[i])) 
       


    for i in range(len(S)):
        
        if(S_flag_Sketch[i] == False):
            continue
        S[i] = Sparse_Sketching.calculate(PPII[i],S[i], PR_UCB.sketch_size)
        R[i] = Sparse_Sketching.calculate(PPII_R[i], R[i], PR_UCB.sketch_size)

        
    for i in range(len(S_hat)):
        if(S_hat_flag_SKetch[i] == False):
            continue

        S_hat[i] = Sparse_Sketching.calculate(PPII_hat[i],S_hat[i], PR_UCB.sketch_size)
        R_hat[i] = Sparse_Sketching.calculate(PPII_R_hat[i],R_hat[i], PR_UCB.sketch_size)
    
    
    # update parameter
    for a in config.A:

        # PR_UCB.Fai[a] += np.dot(S[a].T, S[a])
        # PR_UCB.B_aj[a] += np.dot(S[a].T, R[a])
        PR_UCB.Fai[a] += Sparse_Sketching.matr_multiply(S[a].T, S[a])
        PR_UCB.B_aj[a] += Sparse_Sketching.matr_multiply(S[a].T, R[a])

        # PR_UCB.Fai_hat[a] = PR_UCB.eta * PR_UCB.Fai_hat[a] + np.dot(S_hat[a].T, S_hat[a])
        # PR_UCB.B_aj_hat[a] = PR_UCB.eta * PR_UCB.B_aj_hat[a] + np.dot(S_hat[a].T, R_hat[a])
        PR_UCB.Fai_hat[a] = PR_UCB.eta * PR_UCB.Fai_hat[a] + Sparse_Sketching.matr_multiply(S_hat[a].T, S_hat[a])
        PR_UCB.B_aj_hat[a] = PR_UCB.eta * PR_UCB.B_aj_hat[a] + Sparse_Sketching.matr_multiply(S_hat[a].T, R_hat[a])


        PR_UCB.psi[a] = np.identity(_d) + PR_UCB.Fai[a] + PR_UCB.gamma * PR_UCB.Fai_hat[a]
        t1 = np.linalg.inv(PR_UCB.psi[a].astype(np.float))
        t2 = PR_UCB.B_aj[a] + PR_UCB.gamma * PR_UCB.B_aj_hat[a]
        # PR_UCB.Theta[a] = np.dot(t1, t2)
        PR_UCB.Theta[a] = Sparse_Sketching.matr_multiply(t1, t2)
