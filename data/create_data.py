import random
import numpy as np


np.random.seed(1)
random.seed(1)
N = 1400
k = [70, 105, 175, 140, 210] #1400
p = 40     # dimension
sigma_s = 0.2
sigma_c = 0.01
w_c = []
sigma_d = 0.01
w_d = []
time_interval = 0.001
T = time_interval*N
dataset = []
label = []

flag_i = np.full((N),-1)

w_c1 = np.random.normal(0, sigma_c ** 2, (1, p))
w_d1 = np.random.normal(0, sigma_d ** 2, (1, p))
click_1 = []
while len(click_1) < k[0]:
    rnd_i = random.randint(0, N-1)
    if flag_i[rnd_i] == -1:
        click_1.append(rnd_i)
        flag_i[rnd_i] = 0

w_c2 = np.random.normal(-0.2, (2*sigma_c) ** 2, (1, p))
w_d2 = np.random.normal(0.2, (2*sigma_d) ** 2, (1, p))
click_2 = []
while len(click_2) < k[1]:
    rnd_i = random.randint(0, N-1)
    if flag_i[rnd_i] == -1:
        click_2.append(rnd_i)
        flag_i[rnd_i] = 1

w_c3 = np.random.normal(-0.4, (3*sigma_c) ** 2, (1, p))
w_d3 = np.random.normal(0.4, (3*sigma_d) ** 2, (1, p))
click_3 = []
while len(click_3) < k[2]:
    rnd_i = random.randint(0, N-1)
    if flag_i[rnd_i] == -1:
        click_3.append(rnd_i)
        flag_i[rnd_i] = 2

w_c4 = np.random.normal(-0.6, (4*sigma_c) ** 2, (1, p))
w_d4 = np.random.normal(0.6, (4*sigma_d) ** 2, (1, p))
click_4 = []
while len(click_4) < k[3]:
    rnd_i = random.randint(0, N-1)
    if flag_i[rnd_i] == -1:
        click_4.append(rnd_i)
        flag_i[rnd_i] = 3

w_c5 = np.random.normal(-0.8, (5*sigma_c) ** 2, (1, p))
w_d5 = np.random.normal(0.8, (5*sigma_d) ** 2, (1, p))
click_5 = []
while len(click_5) < k[4]:
    rnd_i = random.randint(0, N-1)
    if flag_i[rnd_i] == -1:
        click_5.append(rnd_i)
        flag_i[rnd_i] = 4

cnt = 0
N_click = 0
N_conversion = 0
for i in range(N):
    s = np.random.normal(0.1, sigma_s ** 2, (p, 1))
    c = 0
    cvr = 0.0
    v = 0
    gamma = 0
    if i in click_1:
        c = 1

        cvr = 1 / (1 + np.exp(-np.dot(w_c1, s)))[0][0]

        if random.random() <= cvr:
            v = 1

        lamda_s = np.exp(np.dot(w_d1,s))
        gamma = random.expovariate(lamda_s)

    if i in click_2:
        c = 1
        cvr = 1 / (1 + np.exp(-np.dot(w_c2, s)))[0][0]
        if random.random() <= cvr:
            v = 1

        lamda_s = np.exp(np.dot(w_d2,s))
        gamma = random.expovariate(lamda_s)

    if i in click_3:
        c = 1
        cvr = 1 / (1 + np.exp(-np.dot(w_c3, s)))[0][0]
        if random.random() <= cvr:
            v = 1

        lamda_s = np.exp(np.dot(w_d3,s))
        gamma = random.expovariate(lamda_s)

    if i in click_4:
        c = 1
        cvr = 1 / (1 + np.exp(-np.dot(w_c4, s)))[0][0]
        if random.random() <= cvr:
            v = 1

        lamda_s = np.exp(np.dot(w_d4,s))
        gamma = random.expovariate(lamda_s)

    if i in click_5:
        c = 1
        cvr = 1 / (1 + np.exp(-np.dot(w_c5, s)))[0][0]
        if random.random() <= cvr:
            v = 1

        lamda_s = np.exp(np.dot(w_d5,s))
        gamma = random.expovariate(lamda_s)

    d = 0
    if i*time_interval + gamma <= T:
        d = 1

    y = d*v
    if y == 1:
        cnt += 1
        print(cnt)

    dataset.append([s,c,v,d,gamma])
    label.append(y)

    if c == 1:
        N_click += 1

    if v == 1:
        N_conversion += 1

a = k
while a[0] > 0:
    rnd_i = random.randint(0, N-1)
    if flag_i[rnd_i] == -1:
        flag_i[rnd_i] = 0
        a[0] -= 1
while a[1]  > 0:
    rnd_i = random.randint(0, N-1)
    if flag_i[rnd_i] == -1:
        flag_i[rnd_i] = 1
        a[1] -= 1
while a[2]  > 0:
    rnd_i = random.randint(0, N-1)
    if flag_i[rnd_i] == -1:
        flag_i[rnd_i] = 2
        a[2] -= 1
while a[3]  > 0:
    rnd_i = random.randint(0, N-1)
    if flag_i[rnd_i] == -1:
        flag_i[rnd_i] = 3
        a[3] -= 1
while a[4]  > 0:
    rnd_i = random.randint(0, N-1)
    if flag_i[rnd_i] == -1:
        flag_i[rnd_i] = 4
        a[4] -= 1

data_buffer = []
lamda = 0.01 * N_conversion / N_click
for i in range(N):
    s = dataset[i][0]
    a = flag_i[i]

    # calculate r
    c = dataset[i][1]
    v = dataset[i][2]
    gamma = dataset[i][4]
    y = label[i]

    r_head = 0
    if c == 1:
        r_head = 1

    r_tilde = 0
    if y == 1:
        r_tilde = 1

    e_i = 0
    if c == 1:
        if y == 1:
            e_i = gamma
        else:
            e_i = T - time_interval * i


    r = lamda * r_head + (1 - lamda) * r_tilde

    if type(r) is not float:
        r = float(r)

    if type(e_i) is not float:
        e_i = float(e_i)

    data_buffer.append([s.tolist(),'|',a, r, e_i, y])

f = open('data.txt','w')
for i in data_buffer:
    print(i,file=f)



