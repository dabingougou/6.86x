import numpy as np
from scipy.stats import norm

K = 2 # number of clusters
N = 5 # number of obs
D = 1 # point dimension

mu = np.array([-3, 2])
sigma_sq = np.array([2, 2])
prior = np.array([0.5, 0.5])

x = np.array([0.2, -0.9, -1, 1.2, 1.8])
print(np.dot(prior, mu))

p_ij = np.empty(shape=(N, K))
for j in range(K):
 for i in range(N):
  p_ij[i][j] = norm(mu[j],sigma_sq[j]).pdf(x[i])

p_i = np.zeros(shape=(N))
for i in range(N):
 for j in range(K):
  p_i[i] += prior[j] * p_ij[i][j]

posterior_ji = np.empty(shape=(N, K))
for i in range(N):
    for j in range(K):
        posterior_ji[i][j] = prior[j] * p_ij[i][j] / p_i[i]
print(f'prior:\n{prior}\n')
print(f'space:\n{p_i}\n')
print(f'prob obs i conditional on cluster j:\n{p_ij}\n')
print(f'posterior prob cluster j given obs i:\n{posterior_ji}\n')

p_j_hat = np.zeros(shape=(K))

p_j_hat = np.sum(posterior_ji, axis=0) * (1 / N)

mu_j_hat = (np.dot(x, posterior_ji)) / np.sum(posterior_ji, axis=0)


x_moment2 = np.empty(shape=(N,K))
for i in range(N):
    for j in range(K):
        x_moment2[i][j] = (np.linalg.norm(x[i] - mu_j_hat[j])) ** 2


sigma_sq_j_hat = np.sum(np.multiply(x_moment2, posterior_ji), axis=0) / (D * np.sum(posterior_ji, axis=0))

print(f'n_j_hat:\n{p_j_hat}\n')
print(f'mu_j_hat:\n{mu_j_hat}\n')
print(f'x 2nd moment about mean:\n{x_moment2}\n')
print(f'sigma_sq_j_hat:\n{sigma_sq_j_hat}\n')




