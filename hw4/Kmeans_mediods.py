import numpy as np
K = 2

x = np.array([[0, -6],
              [4, 4],
              [0, 0],
              [-5, 2]])

D = x.shape[1]
N = x.shape[0]

z = np.array([x[3], x[0]])

x_sub_z = np.empty(shape=(D, N, K))
for j in range(z.shape[0]):
    for i in range(x.shape[0]):
        x_sub_z[:, i, j] = x[i] - z[j]

#l1_norm = x - z
print(z)
print(f'x vector minus z vector:\n{x_sub_z}\n')
#print(x_sub_z[2, 0, :])