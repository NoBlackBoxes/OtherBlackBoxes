import numpy as np
import jax.numpy as jnp
from jax import grad, jit
import matplotlib.pyplot as plt
import time

# Specify paths
repo_path = '/home/kampff/NoBlackBoxes/repos/OtherBlackBoxes'
data_path = repo_path + '/boxes/ai/sgd/_data/complex.csv'

# Load data
data = np.genfromtxt(data_path, delimiter=',')
x = data[:,0]
y = data[:,1]

# Inital parameter guesses
A = np.random.rand(1)[0] - 0.5
B = np.random.rand(1)[0] - 0.5
C = np.random.rand(1)[0] - 0.5
D = np.random.rand(1)[0] - 0.5
E = np.random.rand(1)[0] - 0.5
params = jnp.array([A,B,C,D,E])
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
def loss(x, y, params):
    guess = func(x, params)
    err = y - guess
    return jnp.mean(err*err)

# Compute gradient (w.r.t. parameters)
grad_loss = grad(loss, argnums=2)
grad_loss_jit = jit(grad_loss)

# Train
alpha = .000001
beta = 0.99
deltas = np.zeros(num_params)
gradients = np.zeros(num_params)
report_interval = 100
losses = []
start_time = time.time()
num_epochs = 30000
initial_loss = loss(x, y, params)
for i in range(num_epochs):

    # Compute gradient
    gradients = grad_loss_jit(x, y, params)

    # Update delta
    deltas = (alpha * gradients) + (beta * deltas)

    # Update parameters
    params -= (deltas)

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
