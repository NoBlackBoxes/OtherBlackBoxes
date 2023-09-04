import numpy as np
import matplotlib.pyplot as plt
import time

# Specify paths
repo_path = '/home/kampff/NoBlackBoxes/repos/OtherBlackBoxes'
data_path = repo_path + '/boxes/ai/sgd/_data/simple.csv'

# Load data
data = np.genfromtxt(data_path, delimiter=',')
x = data[:,0]
y = data[:,1]

# Inital guesses
A = np.random.rand(1)[0] - 0.5
B = np.random.rand(1)[0] - 0.5
params = np.array([A,B])
num_params = len(params)

# Define function
def func(x, params):
    A = params[0] 
    B = params[1] 
    return (A * x) + B
    
# Define loss
def loss(x, y, params):
    guess = func(x, params)
    err = y - guess
    return np.mean(err*err)

# Define gradient
def grad(x, y, params):
    A = params[0]
    B = params[1] 
    dE_dA = 2 * np.mean(-x * (y - ((A * x) + B)))
    dE_dB = 2 * np.mean(-(y - ((A * x) + B)))
    return np.array([dE_dA, dE_dB])

# Train
alpha = .001
gradients = np.zeros(num_params)
report_interval = 100
losses = []
start_time = time.time()
num_epochs = 3000
initial_loss = loss(x, y, params)
for i in range(num_epochs):

    # Compute gradient
    gradients = grad(x, y, params)

    # Update parameters
    params -= (gradients * alpha)

    # Compute loss
    current_loss = loss(x, y, params)

    # Store MSE
    losses.append(current_loss)
    
    # Report?
    if((i % report_interval) == 0):
        np.set_printoptions(precision=3)
        print("MSE: {0:.2f}, Params: {1}".format(current_loss, params))
end_time = time.time()

# Benchmark
elapsed = end_time - start_time
print("Time per iteration: {0} ms".format(1000*elapsed/num_epochs))

# Compare
prediction = func(x, params)
plt.plot(x, y, 'b.', markersize=1)
plt.plot(x, prediction, 'r.', markersize=1)
plt.show()

plt.plot(np.array(losses))
plt.show()

#FIN
