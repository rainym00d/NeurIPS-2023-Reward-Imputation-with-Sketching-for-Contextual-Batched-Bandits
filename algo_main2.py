'''
this is only for DFM
'''
import random
import numpy as np
import config
import Algo.DFM as Algo_DFM
from tqdm import tqdm

np.random.seed(1)
random.seed(1)

def count_reward(D_i):
    s=0
    for _,_,r,_,_ in D_i:
        s += r
    # print(s)

def Algo_main(D, B, N, _D,_algo=None, mode='', modify='no'):
    pai = np.full((config.M, 1), 1 / config.M)
    # record for reward, click and conversion
    REWARD = []
    C_sum_record = [0]
    V_sum_record = [0]
    for n in tqdm(range(N)):
        C_sum = 0
        # Action = []
        sample_index = np.random.choice(len(D), B)
        D_n = D[sample_index, :]
        if mode == 'DFM':
            X, Y, timestamp = Algo_DFM.data_process(D_n)
            for i in range(config.M):
                DFM = Algo_DFM.train_dfm(X[i], Y[i], timestamp[i], _algo.W_C[i], _algo.W_D[i], continuous=_algo.continuous)
                _algo.W_C[i] = DFM.coef_[:config.d]
                _algo.W_D[i] = DFM.coef_[config.d:]

        N_conv = 0
        N_click = 0
        D_temp = []
        V_record = [] 
        for b in range(B):
            s_Bn_b = np.random.normal(0.1, 0.2 ** 2, (config.d, 1))

            if mode == 'DFM':
                cvr_list = [1 / (1 + np.exp(-np.dot(_algo.W_C[i], s_Bn_b))) for i in config.A]
                a = cvr_list.index(max(cvr_list))

            c = np.random.choice([0, 1], p=[1 - _D.CLICK_prob[a], _D.CLICK_prob[a]])

            if c == 0:
                cvr = 0
                v = 0
                gamma = _D.T
            else:
                C_sum += 1
                cvr = 1 / (1 + np.exp(-np.dot(_D.w_c[a], s_Bn_b)))[0][0]
                v = np.random.choice([0, 1], p=[1 - cvr, cvr])
                lamda_s = np.exp(np.dot(_D.w_d[a], s_Bn_b))
                gamma = random.expovariate(lamda_s)

            d = 0
            if b * _D.time_interval + gamma <= _D.T:
                d = 1

            y = d * v

            e_i = 0
            if c == 1:
                if y == 1:
                    e_i = gamma
                else:
                    e_i = _D.T - _D.time_interval * b

            if c == 1:
                N_click += 1
            if v == 1:
                N_conv += 1

            V_record.append(v)
            D_temp.append([a, s_Bn_b.T[0].tolist(), c, y, e_i])

        r_sum = 0
        lamda = 0.01 * N_conv / N_click
        for i in range(B):
            _, _, c, y, _ = D_temp[i]
            v = V_record[i]

            r_head = 0
            if c == 1:
                r_head = 1
            r_wave = 0
            if v == 1:
                r_wave = 1
            r_online = lamda * r_head + (1 - lamda) * r_wave
            r_sum += r_online

            if y == 0:
                r_wave = 0
            r = lamda * r_head + (1 - lamda) * r_wave

            e_i = D_temp[i].pop()
            y = D_temp[i].pop()
            c = D_temp[i].pop()
            D_temp[i].append(r)
            D_temp[i].append(e_i)
            D_temp[i].append(y)

        REWARD.append(r_sum)

        D_temp = np.array(D_temp)
        if len(D) < config.data_size:
            if config.data_buffer_counter + B < config.data_size:
                np.append(D, D_temp, axis=0)
            else:
                np.append(D, D_temp[:config.data_size - config.data_buffer_counter], axis=0)
                D[:B + config.data_buffer_counter - config.data_size] = D_temp[
                                                                        config.data_size - config.data_buffer_counter:]
        elif config.data_buffer_counter + B > config.data_size:
            D[config.data_buffer_counter:] = D_temp[0:config.data_size - config.data_buffer_counter]
            D[:B + config.data_buffer_counter - config.data_size] = D_temp[
                                                                    config.data_size - config.data_buffer_counter:]
        else:
            D[config.data_buffer_counter:config.data_buffer_counter + B] = D_temp
        config.data_buffer_counter += B
        config.data_buffer_counter %= config.data_size


        # CVR: V==1/C==1
        # CTCVR: V==1/all
        C_sum_record.append(C_sum_record[-1] + C_sum)
        V_sum_record.append(V_sum_record[-1] + sum(V_record))

    return REWARD, C_sum_record, V_sum_record

if __name__ == '__main__':
    mode = 'DFM'
    _D_init = config.Data_config('linear')
    
    _algo = config.DFM_config()

    D_init = config.load_data()

    mu = 1.0
    w = 1.8
    mode = 'DFM'
    modify = "no"
    D = D_init
    _D = _D_init
    _algo = config.DFM_config()
    _algo.continuous = 1
    reward, c, v = Algo_main(D_init, config.B, config.N, _D_init, _algo, mode, modify=modify)
    print(c)
    print(v)
