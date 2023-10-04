import numpy as np
import config


def linear_reward(theta, s):
    return np.dot(theta, s)


def exponential_reward(theta, s):
    G_E_gradient = np.exp(np.dot(theta, s))*s
    return np.dot(theta, G_E_gradient)


def polynomial_reward(theta, s):
    G_E_gradient = 2 * np.dot(theta, s) * s
    return np.dot(theta, G_E_gradient)


def kernel_s(s):
    # D_kernel must be even number
    D = config.D_kernel // 2
    # parameter: sigma; range: [-12: +2: 12]
    sigma = 2 ** -(-12 + 1) / 2
    u_i = None
    U = []
    for i in range(D):
        u_i = np.random.normal(0, sigma ** -2, (1, config.d))
        U.append(u_i)
    
    u_cos = np.cos(np.dot(U, s)).reshape((-1, 1))
    u_sin = np.sin(np.dot(U, s)).reshape((-1, 1))
    u_all = np.concatenate((u_cos, u_sin))
    fai_s = 1 / np.sqrt(D) * u_all

    return fai_s


def kernelized_reward(theta, s):
    return np.dot(theta, s)


REWARD_DICT = {
    'linear': linear_reward, 
    'exp': exponential_reward,
    'poly': polynomial_reward,
    'kernel': kernelized_reward,
}