import numpy as np
import pandas as pd
import config
from Algo import SBUCB
from Algo import PUIR
from Algo import SPUIR
from Algo import EXP3S1
from Algo import BEXP3S1_IPW
from Algo import BLTS_B
from tqdm import trange
import sys
import reward
import nni
import warnings
warnings.filterwarnings("ignore")

# name of algorithm
ALGO_LIST = [
    'SBUCB', 
    'PUIR', 
    'SPUIR', 
    'PUIR-RS', 
    'SPUIR-RS', 
    'EXP3S1',
    'BLTS-B',
    'BEXP3S1-IPW'
]


def Algo_main(D, B, N, _D, _UCB, ALGO, REWARD):
    '''
    N：number of episodes
    D :data
    _D : data_config
    _UCB: config of algorithm
    ALGO: name of algorithm
    B:buffer_size
    REWARD: type of reward
    '''
    

    Reward_sum = [] # reward of each episode
    cvr_record = [] # cvr of each episode
    ctcvr_record = []   # cvctr of each episode

    # reward function
    cal_reward = reward.REWARD_DICT[REWARD]

    for n in trange(N):
        # offline update

        D_n = D[0:B, :]

        if(ALGO == "SBUCB"):
            SBUCB.UCB(D_n, _UCB, REWARD)
        if(ALGO == 'PUIR'):
            PUIR.Algo_PRUCB(D_n, _UCB, REWARD)
        if(ALGO == 'SPUIR'):
            SPUIR.Algo_PRUCB(D_n, _UCB, REWARD)
        if(ALGO == 'PUIR-RS'):
            step = N//10
            _UCB.gamma = (n//step+1)*0.1
            PUIR.Algo_PRUCB(D_n, _UCB, REWARD)
        if(ALGO == 'SPUIR-RS'):
            step = N//10
            _UCB.gamma = (n//step+1)*0.1
            SPUIR.Algo_PRUCB(D_n, _UCB, REWARD)
        if(ALGO == 'EXP3S1'):
            EXP3S1.EXP3S_1(D_n, n, _UCB)
        if(ALGO == 'BEXP3S1-IPW'):
            BEXP3S1_IPW.BEXP3S_1_IPW(D_n, n, _UCB)
        if (ALGO == 'BLTS-B'):
            BLTS_B.BLTS_B(D_n, _UCB)
        
        # online recommendation
        N_conv = 0
        N_click = 0

        click_list = []
        conversion_list = []
        D_temp = []

        # To avoid repeating computation of np.linalg.inv & .T 
        # & <Theta, s> & sqrt(s*Fai^-1*s.T)
        inv_of_Fai = []
        Theta_T = []
        for a_j in config.A:
            Theta_T.append(_UCB.Theta[a_j].T)
            inv_of_Fai.append(np.linalg.inv(_UCB.Fai[a_j]))
        linear_reward = [] # for offline: tau = <theta_A, s>
        upper_confidence_bound = [] # for offline: C = sqrt(s*Fai^-1*s)

        # generate click,cvr,v and conversion
        # calculate lambda
        for b in range(B):
            # simulate env
            _s_b = np.random.normal(0.1, 0.2 ** 2, (config.d, 1))
            tmp_LR = [] 
            tmp_UCB = []
            s_b = _s_b
            if REWARD == 'kernel':
                s_b = reward.kernel_s(s_b).reshape((-1, 1))
            # choose arm
            if(ALGO == "SBUCB"):
                pai = []
                for a_j in config.A:
                    t1 = cal_reward(Theta_T[a_j], s_b) 
                    tmp_LR.append(t1)
                    t2 = np.dot(s_b.T, inv_of_Fai[a_j])
                    t3 = np.sqrt(np.dot(t2, s_b))
                    tmp_UCB.append(t3)
                    pai.append(t1 + _UCB.miu_ucb*t3)
                a = np.argmax(pai)
            if(ALGO == 'PUIR' or ALGO == 'SPUIR' or ALGO == 'PUIR-RS' or ALGO == 'SPUIR-RS'):
                pai = []
                for a_j in config.A:
                    t1 = cal_reward(Theta_T[a_j], s_b) 
                    tmp_LR.append(t1)
                    t2 = np.dot(s_b.T, inv_of_Fai[a_j])
                    t3 = np.sqrt(np.dot(t2, s_b))
                    tmp_UCB.append(t3)
                    pai.append(t1 + _UCB.alpha*t3)
                a = np.argmax(pai)
            if(ALGO == 'EXP3S1'):
                try:
                    a = np.random.choice(config.A, p=np.reshape(pai, (config.M,)))
                except:
                    pai = np.full((config.M, 1), 1 / config.M)
                    a = np.random.choice(config.A, p=np.reshape(pai, (config.M,)))
                finally:
                    pai = EXP3S1.EXP3S_2(s=s_b, a=a, n=n, _EXP3S=_UCB)
                    a = np.random.choice(config.A, p=np.reshape(pai, (config.M,)))
            if(ALGO == 'BEXP3S1-IPW'):
                try:
                    a = np.random.choice(config.A, p=np.reshape(pai, (config.M,)))
                except:
                    pai = np.full((config.M, 1), 1 / config.M)
                    a = np.random.choice(config.A, p=np.reshape(pai, (config.M,)))
                finally:
                    pai = BEXP3S1_IPW.BEXP3S_2_IPW(s=s_b, a=a, n=n, _EXP3S=_UCB)
                    a = np.random.choice(config.A, p=np.reshape(pai, (config.M,)))
            if (ALGO == 'BLTS-B'):
                pai = []
                for a_j in config.A:
                    t = np.dot(Theta_T[a_j], s_b)
                    pai.append(t)
                a = np.argmax(pai)
            
            click = np.random.choice([0, 1], p=[0.5, 0.5])

            cvr = 1 / (1 + np.exp(-np.dot(_D.w_c[a], s_b)))[0][0]

            # conversion
            conversion = 0
            if click == 1:
                conversion = np.random.choice([0, 1], p=[1-cvr, cvr])

            click_list.append(click)
            conversion_list.append(conversion)

            if click == 1:
                N_click += 1
            if conversion == 1:
                N_conv += 1

            D_temp.append([a, _s_b.T[0].tolist()])
            if(ALGO == 'PUIR' or ALGO == 'SPUIR' or ALGO == 'PUIR-RS' or ALGO == 'SPUIR-RS' or ALGO == 'SBUCB'):
                linear_reward.append(tmp_LR)
                upper_confidence_bound.append(tmp_UCB)
        # calculate reward
        r = 0
        r_online = 0

        for i in range(B):

            lamda = 0.01 * N_conv / N_click

            r_head = 0
            if click_list[i] == 1:
                r_head = 1

            r_wave = 0
            if conversion_list[i] == 1:
                r_wave = 1


            r = lamda * r_head + (1 - lamda) * r_wave
            r_online += lamda * r_head + (1 - lamda) * r_wave
            D_temp[i].append(r)

            # record linear reward and upper confidence bound for offline learning to avoid repeated computation
            if(ALGO == 'PUIR' or ALGO == 'SPUIR' or ALGO == 'PUIR-RS' or ALGO == 'SPUIR-RS' or ALGO == 'SBUCB'):
                D_temp[i].append(linear_reward[i])
                D_temp[i].append(upper_confidence_bound[i])
            else:
                D_temp[i].append(0)
                D_temp[i].append(0)

        Reward_sum.append(r_online)
        cvr_record.append(N_conv / N_click)
        ctcvr_record.append(N_conv / B)
        # for next offline training
        D_temp = np.array(D_temp, dtype=object)
        D[0:B, :] = D_temp

        nni.report_intermediate_result(sum(Reward_sum) / B / len(Reward_sum))

    cvr_ctcvr = {'cvr':cvr_record, 'ctcvr':ctcvr_record}
    return pd.DataFrame(Reward_sum), pd.DataFrame(cvr_ctcvr)

def generate_default_parameters(ALGO):
    '''
    return default parameters in dict
    '''
    if(ALGO == 'PUIR'):
        return {"alpha": 0.1, "eta": 0.1, "gamma": 0.1}
    elif(ALGO == 'SPUIR'):
        return {"alpha": 0.2, "eta": 0.9, "gamma": 0.1, "sketch_size":150, "buffer_size":1400}
    elif(ALGO == 'PUIR-RS'):
        return {"alpha": 0.1, "eta": 0.1}
    elif(ALGO == 'SPUIR-RS'):
        return {"alpha": 0.1, "eta": 0.1, "sketch_size":150, "buffer_size":1400}
    elif(ALGO == 'BEXP3S1-IPW' or ALGO == 'EXP3S1'):
        return dict() # no need for parameter
    elif (ALGO == 'BLTS-B'):
            return {"miu": 0.25, "gamma": 0.2}
    return {}

if __name__ == '__main__':
    argv = sys.argv
    
    ALGO = str(argv[1])
    if(ALGO not in ALGO_LIST):
        print('algorithm name error!')
        exit(0)

    REWARD = 'linear'
    if(REWARD not in reward.REWARD_DICT):
        print('reward fuction name error!')
        exit(0)
    
    RECEIVED_PARAMS = nni.get_next_parameter()
    PARAMS = generate_default_parameters(ALGO)
    PARAMS.update(RECEIVED_PARAMS)

    B = config.B
    Algo_config = None
    if(ALGO == "SBUCB"):
        Algo_config = config.UCB_config(REWARD)
        Algo_config.miu_ucb = 1.00
    if(ALGO == 'PUIR'):
        Algo_config = config.PR_UCB_config(REWARD)
        Algo_config.alpha = PARAMS['alpha']
        Algo_config.eta = PARAMS['eta']
        Algo_config.gamma = PARAMS['gamma']
    if(ALGO == 'SPUIR'):
        Algo_config = config.PR_UCB_config(REWARD)
        Algo_config.alpha = PARAMS['alpha']
        Algo_config.eta = PARAMS['eta']
        Algo_config.gamma = PARAMS['gamma']
        Algo_config.sketch_size = int(PARAMS['sketch_size'])
        B = int(PARAMS['buffer_size'])
    if(ALGO == 'PUIR-RS'):
        Algo_config = config.PR_UCB_config(REWARD)
        Algo_config.alpha = PARAMS['alpha']
        Algo_config.eta = PARAMS['eta']
    if(ALGO == 'SPUIR-RS'):
        Algo_config = config.PR_UCB_config(REWARD)
        Algo_config.alpha = PARAMS['alpha']
        Algo_config.eta = PARAMS['eta']
        Algo_config.sketch_size = int(PARAMS['sketch_size'])
        B = int(PARAMS['buffer_size'])
    if(ALGO == 'EXP3S1'):
        Algo_config = config.EXP3S1_config()
    if(ALGO == 'BEXP3S1-IPW'):
        Algo_config = config.BEXP3S1_IPW_config()
    if (ALGO == 'BLTS-B'):
        Algo_config = config.BLTS_B_config()
        Algo_config.miu = PARAMS['miu']
        Algo_config.gamma = PARAMS['gamma']


    D = config.load_data()
    _D = config.Data_config(REWARD)
    
    reward, cvr_ctcvr = Algo_main(D, B, config.N, _D, Algo_config,  ALGO=ALGO, REWARD=REWARD)

    sum_reward = reward[0].sum() / config.N / B
    nni.report_final_result(float(sum_reward))
