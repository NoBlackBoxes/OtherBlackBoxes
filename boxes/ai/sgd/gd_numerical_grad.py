from tracemalloc import start
import numpy as np
import matplotlib.pyplot as plt
import time
from torch import le

# Specify paths
repo_path = '/home/kampff/NoBlackBoxes/repos/OtherBlackBoxes'
data_path = repo_path + '/boxes/ai/sgd/_data/noisy.csv'

# Load data
data = np.genfromtxt(data_path, delimiter=',')
x = data[:,0]
y = data[:,1]

# Inital guesses
A = np.random.rand(1)[0] - 0.5
B = np.random.rand(1)[0] - 0.5
C = np.random.rand(1)[0] - 0.5
D = np.random.rand(1)[0] - 0.5
E = np.random.rand(1)[0] - 0.5
params = np.array([A,B,C,D,E])
num_params = len(params)

# Define function
def func(x, params):
    A = params[0] 
    B = params[1] 
    C = params[2] 
    D = params[3] 
    E = params[4] 
    return A * (x**5) + B * (x**4) + C * (x**3) + D * (x**2) + E * (x)
    
# Define loss
def mse(x, y, params):
    guess = func(x, params)
    err = y - guess
    return np.mean(err*err)

# Train
initial_mse = mse(x, y, params)
last_error = initial_mse
step_size = 0.00001
alpha = 1
gradients = np.zeros(num_params)
report_interval = 100
mses = []
start_time = time.time()
num_epochs = 30000
for i in range(num_epochs):
    for p in range(num_params):
        guess_params = np.copy(params)
        guess_params[p] += step_size
        e0 = mse(x, y, params)
        e1 = mse(x, y, guess_params)
        grad = e0 - e1

        # Update
        params[p] += (grad * alpha)

        # Store previous gradient
        gradients[p] = grad

    # Store MSE
    mses.append(e0)
    
    # Report?
    if((i % report_interval) == 0):
        np.set_printoptions(precision=3)
        print("MSE: {0:.2f}, Params: {1}".format(e0, params))
end_time = time.time()

# Benchmark
elapsed = end_time - start_time
print("Time per iteration: {0} ms".format(1000*elapsed/num_epochs))

# Compare
prediction = func(x, params)
plt.plot(x, y, 'b.', markersize=1)
plt.plot(x, prediction, 'r.', markersize=1)
plt.show()

plt.plot(np.array(mses))
plt.show()

#FIN
