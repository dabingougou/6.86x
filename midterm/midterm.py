import numpy as np

# input
z = np.array([[0, 0, -1],
             [2, 0, -1],
             [3, 0, -1],
             [0, 2, -1],
             [2, 2, -1],
             [5, 1, 1],
             [5, 2, 1],
             [2, 4, 1],
             [4, 4, 1],
             [5, 5, 1]])
x = np.array([[0, 0],
             [2, 0],
             [3, 0],
             [0, 2],
             [2, 2],
             [5, 1],
             [5, 2],
             [2, 4],
             [4, 4],
             [5, 5]])
y = np.array([[-1],[-1],[-1],[-1],[-1],[1],[1],[1],[1],[1]])

# theta = np.zeros(2)
# theta0 = np.zeros(1)
theta = np.array([4, 4])
theta0 = np.array([-18])

n_mistakes = np.zeros(10)

sigma = np.ones(2)
sigma0 = np.ones(1)

#a = np.array([1, 3])
#b = np.array([[2], [4]])
#print(np.dot(a, b))
loss = np.zeros(10)
minloss = -999
T = 0

print(np.arange(z.shape[0]))
r = np.array([6, 9, 1, 5, 2, 7, 3, 8, 4, 0])
print(z[0:6, 2])
#print(r)
#np.random.shuffle(r)
# for i in r:

while minloss <= 0:
#for T in range(20):
    for i in r:
#range(len(y)):
        print(f'In loop {T + 1}:\n')
        print(f'training point: {i+1}\n')
        loss[i] = y[i] * (np.dot(theta, x[i]) + theta0)
        if loss[i] <= 0:
            theta = theta + 0.01 * y[i] * x[i]  #.ravel()
            theta0 = theta0 + 1
            n_mistakes[i] = n_mistakes[i] + 1
    loss[i] = y[i] * (np.dot(x[i], theta) + theta0)
    minloss = np.min(loss)
    T = T + 1
print(f'{T} loops\n')
print(f'minloss: {minloss}')
print(f'Mistakes: {n_mistakes}')
print(f'Theta: {theta}\n')
print(f'Theta0: {theta0}\n')
print(f'Error: {loss}\n')

theta = np.array([0.5, 0.5])
theta0 = np.array([-2.5])
agreement = np.zeros(len(y))
single_loss = np.zeros(len(y))
hinge_loss = 0
print(f'Theta: {theta}\nTheta0: {theta0}\n')
for i in range(len(y)):
    agreement[i] = y[i] * (np.dot(theta, x[i]) + theta0)
    single_loss[i] = np.maximum((1 - agreement[i]), 0)
    print(f'single loss point {i+1}:\n{single_loss}\n')
    hinge_loss += single_loss[i]
    print(f'total loss at {i + 1}:\n{hinge_loss}\n')


