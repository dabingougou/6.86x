import numpy as np

x = np.array([[0, 0],
              [2, 0],
              [1, 1],
              [0, 2],
              [3, 3],
              [4, 1],
              [5, 2],
              [1, 4],
              [4, 4],
              [5, 5]])

y = np.array([-1,-1,-1,-1,-1,1,1,1,1,1])

print(y.shape)

alpha = np.array([1, 65, 11, 31, 72, 30, 0, 21, 4, 15])

phi = np.zeros((10,3))
#print(x[9, 1])
for i in range(len(y)):
    phi[i] = np.array([x[i,0]**2, np.sqrt(2) * x[i, 0] * x[i, 1], x[i, 1]**2])
print(f'phi:\n{phi}')

theta = np.zeros(3)
theta0 = np.zeros(1)
eye = np.eye(10)
eye3 = np.eye(10) * 3
#print(eye3)

print(np.dot(eye[9], phi))
#print(np.multiply(alpha, y))
summer = np.multiply(alpha, y)
#print(summer)
theta = np.dot(summer, phi)
print(f'theta:\n{theta}\n')
theta0 = np.dot(alpha, y)
print(f'theta0:\n{theta0}\n')

agreement = np.zeros(10)
for i in range(len(y)):
    agreement[i] = y[i] * (np.dot(theta, phi[i]) + theta0)
print(f'Agreement:\n{agreement}\n')