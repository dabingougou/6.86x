import numpy as np
K = 2

x = np.array([[0, -6],
              [4, 4],
              [0, 0],
              [-5, 2]])

D = x.shape[1]

z = np.array([x[3], x[0]])

l1_norm = x - D
print(z)
print(l1_norm)