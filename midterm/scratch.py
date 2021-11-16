import numpy as np

x = np.array([[1, 2], [3, 4]])
theta = np.array([5, 6])
theta2 = np.array([[5], [6]])
theta3 = theta2.ravel()

# print(theta2)
# print(theta3)
# print(np.dot(x[1], theta))
# print(x[1] + theta)
print(x[0].ravel())
