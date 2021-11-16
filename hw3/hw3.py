# HW3
import numpy as np

x = np.random.randint(1, 10, size=(3, 4))
print(x)
w = np.array([[1, 0, -1],
             [0, 1, -1],
             [-1, 0, -1],
              [0, -1, -1]])
print(f'\nw matrix:\n {w}')

v = np.array([[1, 1, 1, 1, 0],
             [-1, -1, -1, -1, 2]])
print(f'\nw matrix:\n {v}')

x = np.array([3, 14, 1])
print(f'\nx:\n{x}\n')

# First layer linear transformation
z = np.matmul(w, x)

print(f'\nfirst layer units (z):\n{z}\n')

f1 = np.maximum(np.zeros(z.shape[0]), z)
f1 = np.append(f1, 1)
print(f'\nfirst layer activation (f1):\n{f1}\n')

u = np.matmul(v, f1)
print(f'second layer transformation (u):\n{u}\n')
u_relu = np.maximum(np.zeros(u.shape[0]), u)

# Implementing softmax
blunter = np.max(u_relu)
u1 = u_relu - blunter
print(f'blunter:\n{blunter}\n')
print(f'ReLU u (u_relu):\n{u_relu}\n')
print(f'adjusted u (u1):\n{u1}\n')

expu = np.exp(np.array([3, 0]))
sum_expu = np.sum(expu)
print(f'exponentials of adjusted u:\n{expu}\n')
print(f'sum of exponentials:\n{sum_expu}\n')

p = expu / sum_expu
print(f'Probability distribution:\n{p}\n')




# LSTM
x = np.array([0, 1, 1, 0, 1, 1])
w_ih = 0;       w_fh = 0;   w_ch = -100;    w_oh = 0;
w_ix = 100;     w_fx = 0;   w_cx = 50;      w_ox = 100;
b_i = 100;      b_f = -100; b_c = 0;        b_o = 0;
h = np.zeros(len(x))
c = np.zeros(len(x))
for i in range(1, len(x)):
    igate = round(1 / (1 + np.exp(- w_ih * h[i-1] - w_ix * x[i] - b_i)))
    fgate = round(1 / (1 + np.exp(- w_fh * h[i-1] - w_fx * x[i] - b_f)))
    cell = w_ch * h[i-1] + w_cx * x[i] + b_c
    ogate = round(1 / (1 + np.exp(- w_oh * h[i-1] - w_ox * x[i] - b_o)))
    c[i] = fgate * c[i-1] + igate * round(np.tanh(cell))
    h[i] = np.rint(ogate * round(np.tanh(c[i])))
print(f'h:\n{h}\n')
print(f'c:\n{c}\n')

sigma = 1 / (1 + np.exp(1.15))
print(f'sigma: {sigma}')
loss = (1 / 2) * ((0.24048908305088898 - 1) ** 2)
print(f'loss: {loss}')
sigma_prime = np.exp(-(-1.15)) / ((1 + np.exp(-(-1.15))) ** 2)
pd_w1 = (sigma - 1) * sigma_prime * (-5) * 1 * 3
print(f'pd_w1: {pd_w1}')

pd_w2 = (sigma - 1) * sigma_prime * 0.03
pd_b  = (sigma - 1) * sigma_prime
print(f'pd_w2: {pd_w2}')
print(f'pd_b: {pd_b}')
