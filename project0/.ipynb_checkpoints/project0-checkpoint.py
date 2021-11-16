import numpy as np

def randomization(n):
    """takes positive integer n and returns A, a random n by 1 numpy array"""
    val = np.random.random([n, 1])
    return val
print(randomization(5))

print("This is the end of problem 1")

def operations(h, w):
    """Generate matrices A, B, and s,
    of dimension h by w, where s = A + B"""
    A = np.random.random([h, w])
    B = np.random.random([h, w])
    s = A + B
#    t = (A, B, s)
    return A, B, s 

print(operations(3, 2))

print("This is the end of problem 2")

A = np.random.random([1, 1])
B = np.random.random([1, 1])

def norm(A, B):
    """
    Takes two numpy column arrays A and B,
    add them,
    then compute the L2 norm
    """
#    absum = A + B
    s = np.linalg.norm(A + B)
    return s

print(norm(A, B))

print("This is the end of problem 3")

inputs = np.array([3, 1])
weights = np.random.random([2, 1])
# print(weights.transpose())
# print(np.matmul(weights.transpose(), inputs))
def neural_network(inputs, weights):
    """
    Takes in inputs and weights, 
    which are numpy arrays of shape (2, 1),
    to produce a numpy array of shape (1, 1), 
    assuming the tanh activation
    """
    z = np.tanh(np.matmul(weights.transpose(), inputs))
    return z
print(neural_network(inputs, weights))
print("This is the end of problem 4")

def scalar_function(x, y):
    """
    Scalar production 
    """
    if x <= y :
        out = x * y
    else : 
        out = x / y
    return out
print(scalar_function(6, 7))
print("This is the end of problem 5")


x = np.array([3, 7])
y = np.array([11, 1])
def vector_function(x, y):
    """
    This time x and y will be vectors of size dimension
    """
    out = np.vectorize(scalar_function) 
    return out(x, y)
print(vector_function(x, y))
print("This is the end of problem 6")

