import numpy as np
import jax.numpy as jnp
from jax import grad, jit, nn
import matplotlib.pyplot as plt
import time

# Specify paths
repo_path = '/home/kampff/NoBlackBoxes/repos/OtherBlackBoxes'
data_path = repo_path + '/boxes/ai/sgd/_data/complex.csv'

# Load data
data = np.genfromtxt(data_path, delimiter=',')
x = data[:,0]
y = data[:,1]
x = np.expand_dims(x,1)
y = np.expand_dims(y,1)

# Inital parameter guesses
num_neurons = 2
W1 = np.random.rand(num_neurons) - 0.5
B1 = np.random.rand(num_neurons) - 0.5
W1 = np.expand_dims(W1,0)
B1 = np.expand_dims(B1,0)

W2 = np.random.rand(num_neurons) - 0.5
B2 = np.random.rand(num_neurons) - 0.5
W2 = np.expand_dims(W2,0)
B2 = np.expand_dims(B2,0)

# Define function (network)
def func(x, W1, B1, W2, B2):
    hidden = x.dot(W1) + B1
    activations = nn.sigmoid(jnp.sum(hidden, axis=1))
    activations = jnp.expand_dims(activations,1)
    hidden2 = activations.dot(W2) + B2
    output = jnp.sum(hidden2, axis=1)
    return output

# Define loss
def loss(x, y, W1, B1, W2, B2):
    guess = func(x, W1, B1, W2, B2)
    err = np.squeeze(y) - guess
    return jnp.mean(err*err)

# Compute gradient (w.r.t. parameters)
grad_loss_W1 = jit(grad(loss, argnums=2))
grad_loss_B1 = jit(grad(loss, argnums=3))
grad_loss_W2 = jit(grad(loss, argnums=4))
grad_loss_B2 = jit(grad(loss, argnums=5))

# Train
alpha = .001
beta = 0.99
deltas_W1 = np.zeros(num_neurons)
deltas_B1 = np.zeros(num_neurons)
deltas_W2 = np.zeros(num_neurons)
deltas_B2 = np.zeros(num_neurons)
gradients_W1 = np.zeros(num_neurons)
gradients_B1 = np.zeros(num_neurons)
gradients_W2 = np.zeros(num_neurons)
gradients_B2 = np.zeros(num_neurons)
report_interval = 100
losses = []
start_time = time.time()
num_epochs = 3000
initial_loss = loss(x, y, W1, B1, W2, B2)
for i in range(num_epochs):

    # Compute gradients
    gradients_W1 = grad_loss_W1(x, y, W1, B1, W2, B2)
    gradients_B1 = grad_loss_B1(x, y, W1, B1, W2, B2)
    gradients_W2 = grad_loss_W2(x, y, W1, B1, W2, B2)
    gradients_B2 = grad_loss_B2(x, y, W1, B1, W2, B2)

    # Update delta
    deltas_W1 = (alpha * gradients_W1) + (beta * deltas_W1)
    deltas_B1 = (alpha * gradients_B1) + (beta * deltas_B1)
    deltas_W2 = (alpha * gradients_W2) + (beta * deltas_W2)
    deltas_B2 = (alpha * gradients_B2) + (beta * deltas_B2)

    # Update parameters
    W1 -= (deltas_W1)
    B1 -= (deltas_B1)
    W2 -= (deltas_W2)
    B2 -= (deltas_B2)

    # Compute loss
    current_loss = loss(x, y, W1, B1, W2, B2)

    # Store MSE
    losses.append(current_loss)
    
    # Report?
    if((i % report_interval) == 0):
        np.set_printoptions(precision=3)
        print("MSE: {0:.2f}, Weights: {1}".format(current_loss, W2))
end_time = time.time()

# Benchmark
elapsed = end_time - start_time
print("Time per iteration: {0} ms".format(1000*elapsed/num_epochs))

# Compare
prediction = func(x, W1, B1, W2, B2)
plt.plot(x, y, 'b.', markersize=1)
plt.plot(x, prediction, 'r.', markersize=1)
plt.show()

plt.plot(np.array(losses))
plt.show()

#FIN
