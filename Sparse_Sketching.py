import random
import numpy
import math

prime_nums = [9789, 8713, 8641, 8543, 8443, 7603, 7541, 7349, 6211,
             6203, 6199, 5087, 5081, 5077, 4111, 4099, 4093, 3413, 3407, 3391, 2591,
              2441, 2063, 1237, 1231, 877, 607, 353, 283, 223, 211, 199, 179, 101, 
              97, 89, 83, 79, 73, 53, 47, 29, 23, 19, 17, 13, 11, 7, 5, 3, 1]

SS_p = 1 #blocks, eg:1,2,4,6

class hash_family():
    def __init__(self, u, p, s):
        '''
        initialize a,b,c,d,q
        u : domin of input, to decide prime number
        p : number of hash functions
        s : width of one block 
        '''
        self.q = []  # prime number
        self.a = []
        self.b = []
        self.c = []
        self.d = []
        self.s = s  # width of block

        tmp_prime = []
        for prime in prime_nums:
            if prime < 2*u:
                tmp_prime.append(prime)
            if prime < u:
                break

        for i in range(p):
            # q : prime number in [u,2u)
            index = random.choice(range(len(tmp_prime)))
            q = tmp_prime[index]
            
            a = random.choice(range(1, q+1))
            b = random.choice(range(1, q+1))
            c = random.choice(range(1, q+1))
            d = random.choice(range(1, q+1))
            self.q.append(q)
            self.a.append(a)
            self.b.append(b)
            self.c.append(c)
            self.d.append(d)

    def h_func(self, i, r):
        '''
        h(i,r) column i, block r
        '''
        h_i_r = (self.a[r]+self.b[r]*i) % self.q[r] % self.s
        return h_i_r

    def g_func(self, i, r):
        '''
        h(i,r) column i, block r
        '''
        g_i_r = 2*((self.c[r]+self.d[r]*i) % self.q[r] % 2) - 1
        return g_i_r


class Sparse_Transform():
    def __init__(self, c, N_aj, p):
        '''
        c: rows, N_aj: columns, p: blocks
        
        '''
        self.c = c  # rows
        self.N_aj = N_aj  # columns
        self.p = p  # blocks
        self.m = int(c / p)  # rows per block
        self.hash_family = hash_family(N_aj, p, self.m)


    def transform(self, src_mat):
        '''
        src_mat: source matrix
        generate sketching matrix
        '''
        d = src_mat.shape[1]
        res_matrix = numpy.zeros((self.c, d))
        PI_matrix = numpy.zeros((self.c, self.N_aj))
        sqrt_p = 1 / math.sqrt(self.p)
        for r in range(self.p):
            for i in range(self.N_aj):
                res_i = self.hash_family.h_func(i, r) + r*self.m
                PI_matrix[res_i, i] = self.hash_family.g_func(i, r) * sqrt_p
            

        ROW_list = list()
        for i in range(self.c):
            tmp = list()
            for j in range(self.N_aj):
                if(PI_matrix[i,j] > 0):
                    tmp.append( (j,1) )
                if(PI_matrix[i,j] < 0):
                    tmp.append( (j,-1) )
            ROW_list.append(tmp)
        
        return ROW_list

def calculate(ROW_list, src_matrix, SS_c):
    '''
    calculate result of sketching
    ROW_list: sketching matrix
    src_matrix: source matrix
    SS_c: Sketch size c
    '''
    d = src_matrix.shape[1]
    res_matrix = numpy.zeros((SS_c, d))
    for i in range(SS_c):
        tmp = numpy.zeros((1,d))
        for item in ROW_list[i]:
            if(item[1]>0):
                for j in range(d):
                    tmp[0,j] +=  src_matrix[item[0],j]
            if(item[1]<0):
                for j in range(d):
                    tmp[0,j] -=  src_matrix[item[0],j]
        res_matrix[i,:] = tmp
    p_sqrt =  1 / math.sqrt(SS_p)
    for i in range(SS_c):
        for j in range(d):
            res_matrix[i,j] = res_matrix[i,j] * p_sqrt
    return res_matrix

def matr_multiply(matA,matB):
    a = matA.shape[0]
    b = matA.shape[1]
    c = matB.shape[1]
    res = numpy.zeros((a,c))
    
    for i in range(a):
        for j in range(b):
            for k in range(c):
                res[i,k] += matA[i,j] * matB[j,k]
    return res

