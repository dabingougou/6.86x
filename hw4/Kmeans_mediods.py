import numpy as np
# from sklearn_extra.cluster import KMedoids
K = 2

x = np.array([[0, -6],
              [4, 4],
              [0, 0],
              [-5, 2]])

D = x.shape[1]
N = x.shape[0]

z = np.array([x[3], x[0]])

x_sub_z = np.empty(shape=(N, K, D))
for j in range(z.shape[0]):
    for i in range(x.shape[0]):
        x_sub_z[i, j, :] = x[i] - z[j]

l1_norm = np.linalg.norm(x_sub_z, ord=1, axis=2)

# mask = l1_norm==np.max(l1_norm, axis=1)
#l1_norm = x - z
print(f'z:\n{z}\n')
print(f'x vector minus z vector:\n{x_sub_z}\n')
print(f'l1 vector:\n{l1_norm}\n')
# print(f'mask:\n{mask}\n')

#print(np.linalg.norm(x_sub_z[0, 0, :], ord=1, axis=2))
vec = x_sub_z[2, 0, :]
print(vec)
print(l1_norm.argmax(axis=1))

groupings = l1_norm.argmax(axis=1)

# for j in range(K):
#     for i in range(N):
#        if groupings[i] = j
