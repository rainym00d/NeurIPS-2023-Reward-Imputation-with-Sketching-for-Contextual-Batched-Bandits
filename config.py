import random
import numpy as np
import math

np.random.seed(1)
random.seed(1)

M = 5 # number of arms
N = 90 # number of episodes
B = 1400 # batch size
d = 40 # dimension of data
A = [0, 1, 2, 3, 4] # arm
data_size = 10000 
data_buffer_counter = 0
D_kernel = 100

class Data_config():
    def __init__(self, REWARD):
        self.sigma_c = 0.01
        self.sigma_d = 0.01
        self.miu_c = 0
        self.miu_d = 0
        self.w_c = []
        self.w_d = []
        self.CLICK_prob = np.arange(0.3, 0.5, 0.2/M)
        _d = d
        # if reward fuction is kernel, must change dimension of theta
        if REWARD == 'kernel':
            _d = D_kernel
        for i in range(M):
            w_c_i = np.random.normal(self.miu_c, ((i + 1) * self.sigma_c) ** 2, (1, _d))
            w_d_i = np.random.normal(self.miu_d, ((i + 1) * self.sigma_d) ** 2, (1, _d))
            self.miu_c -= 0.2
            self.miu_d += 0.2
            self.w_c.append(w_c_i)
            self.w_d.append(w_d_i)

class UCB_config():
    def __init__(self, REWARD):
        self.miu_ucb = 1.0

        self.Theta = []
        self.Fai = []
        self.B_aj = []
        # if reward fuction is kernel, must change dimension of theta
        _d = d
        if REWARD == 'kernel':
            _d = D_kernel
        for i in range(M):
            theta = np.zeros((_d, 1))
            self.Theta.append(theta)
            fai = np.identity(_d)
            self.Fai.append(fai)
            b = np.zeros((_d, 1))
            self.B_aj.append(b)


class DFM_config():
    def __init__(self):
        self.W_C = []
        self.W_D = []
        self.continuous = 1
        for i in range(M):
            self.W_C.append(np.zeros((d, 1)))
            self.W_D.append(np.zeros((d, 1)))

class PR_UCB_config():
    def __init__(self, REWARD):
        self.alpha = 0.2
        self.beta = 0
        self.eta = 1.0
        self.gamma = 0.1

        self.Theta = []
        self.Fai = []
        self.B_aj = []
        self.B_aj_hat = []
        self.Fai_hat = []
        self.psi = []

        self.sketch_size = 200

        # if reward fuction is kernel, must change dimension of theta
        _d = d
        if REWARD == 'kernel':
            _d = D_kernel
        for i in range(M):
            theta = np.zeros((_d, 1))
            self.Theta.append(theta)
            
            b = np.zeros((_d, 1))
            b_hat = np.zeros((_d, 1))
            self.B_aj.append(b)
            self.B_aj_hat.append(b_hat)

            fai = np.zeros((_d, _d))
            fai_hat = np.zeros((_d, _d))
            self.Fai.append(fai)
            self.Fai_hat.append(fai_hat)

            psi = np.identity(_d)
            self.psi.append(psi)


class EXP3S1_config():
    def __init__(self):
        self.Theta = []
        self.Theta_capital = []
        self.Fai = []
        self.B_aj = []
        for i in range(M):
            theta = np.zeros((d, 1))
            self.Theta.append(theta)
            fai = np.identity(d)
            self.Fai.append(fai)
            b = np.zeros((d, 1))
            self.B_aj.append(b)
            self.Theta_capital.append([])
        self.P = np.ones((M, 1))
        self.Q = np.ones((M, 1))
        self.delta = 0.001
        self.eta = (2 * (1 - self.delta) * math.log(M) / (M * N * B)) ** 0.5

class BEXP3S1_IPW_config():
    def __init__(self):
        self.Theta = []
        self.Theta_capital = []
        self.Fai = []
        self.B_aj = []
        for i in range(M):
            theta = np.zeros((d, 1))
            self.Theta.append(theta)
            fai = np.identity(d)
            self.Fai.append(fai)
            b = np.zeros((d, 1))
            self.B_aj.append(b)
            self.Theta_capital.append([])
        self.P = np.ones((M, 1))
        self.Q = np.ones((M, 1))
        self.delta = 0.001
        self.eta = (2 * (1 - self.delta) * math.log(M) / (M * N * B)) ** 0.5
        self.pai = np.ones((M, 1))

class BLTS_B_config():
    def __init__(self):
        self.miu = 1.0
        self.gamma = 0.2
        self.Theta = []
        self.Theta_line = []
        self.Fai = []
        self.B_aj = []
        # if reward fuction is kernel, must change dimension of theta
        _d = d
        for i in range(M):
            theta = np.zeros((_d, 1))
            self.Theta.append(theta)
            theta_line = np.zeros((_d, 1))
            self.Theta_line.append(theta_line)
            fai = np.identity(_d)
            self.Fai.append(fai)
            b = np.zeros((_d, 1))
            self.B_aj.append(b)


# load data
# {a,s,r,e,y}
def load_data():
    data_buffer = []
    f = open('data/data_0120.txt', 'r')
    lines = f.readlines()
    for line in lines:
        l1 = line.strip('[').strip(']').strip('\n')
        l2 = l1.split('|')

        # generate a and r
        l2_1 = l2[1].strip('\', ').strip(']').split(', ')
        a = int(eval(l2_1[0]))
        r = eval(l2_1[1])
        # generate s
        l2_0 = l2[0].strip('], \'').split('], [')
        s = []
        for item in l2_0:
            s.append(eval(item))

        # these 2 lines used for DFM
        # e = dataset[i][52]
        # y = dataset[i][53]

        # All algorihtms use these two lines except DFM
        e = [0] * M # linear reward for pseudo-reward. At the beginning, Theta = 0, so LR=0
        y = [np.sqrt(np.dot(np.array(s).T, np.array(s)))] * M # Upper confidence bound. At the beginning, Fai = identity, so UCB = <S.T, S>
        
        data_buffer.append([a,s,r,e,y])
    
    return np.array(data_buffer, dtype=object)
