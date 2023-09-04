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
num_neurons = 10
W1 = np.random.rand(num_neurons) - 0.5
B1 = np.random.rand(num_neurons) - 0.5
W1 = np.expand_dims(W1,0)
B1 = np.expand_dims(B1,0)

W2 = np.random.rand(num_neurons*num_neurons).reshape((num_neurons, num_neurons)) - 0.5
B2 = np.random.rand(num_neurons) - 0.5

W3 = np.random.rand(num_neurons) - 0.5
B3 = np.random.rand(num_neurons) - 0.5
W3 = np.expand_dims(W3,0)
B3 = np.expand_dims(B3,0)

# Define function (network)
def func(x, W1, B1, W2, B2, W3, B3):
    hidden = x.dot(W1) + B1
    activations = nn.sigmoid(hidden)
    #hidden2 = activations.dot(W2) + B2
    #activations2 = nn.sigmoid(hidden2)
    hidden3 = activations.dot(W3.T) + B3
    output = jnp.sum(hidden3, axis=1)
    return output

# Define loss
def loss(x, y, W1, B1, W2, B2, W3, B3):
    guess = func(x, W1, B1, W2, B2, W3, B3)
    err = np.squeeze(y) - guess
    return jnp.mean(err*err)

# Compute gradient (w.r.t. parameters)
grad_loss_W1 = jit(grad(loss, argnums=2))
grad_loss_B1 = jit(grad(loss, argnums=3))
grad_loss_W2 = jit(grad(loss, argnums=4))
grad_loss_B2 = jit(grad(loss, argnums=5))
grad_loss_W3 = jit(grad(loss, argnums=6))
grad_loss_B3 = jit(grad(loss, argnums=7))

# Train
alpha = .001
beta = 0.99
deltas_W1 = np.zeros(num_neurons)
deltas_B1 = np.zeros(num_neurons)
deltas_W2 = np.zeros(num_neurons)
deltas_B2 = np.zeros(num_neurons)
deltas_W3 = np.zeros(num_neurons)
deltas_B3 = np.zeros(num_neurons)
report_interval = 100
losses = []
start_time = time.time()
num_epochs = 3000
initial_loss = loss(x, y, W1, B1, W2, B2, W3, B3)
for i in range(num_epochs):

    # Compute gradients
    gradients_W1 = grad_loss_W1(x, y, W1, B1, W2, B2, W3, B3)
    gradients_B1 = grad_loss_B1(x, y, W1, B1, W2, B2, W3, B3)
    gradients_W2 = grad_loss_W2(x, y, W1, B1, W2, B2, W3, B3)
    gradients_B2 = grad_loss_B2(x, y, W1, B1, W2, B2, W3, B3)
    gradients_W3 = grad_loss_W3(x, y, W1, B1, W2, B2, W3, B3)
    gradients_B3 = grad_loss_B3(x, y, W1, B1, W2, B2, W3, B3)

    # Update delta
    deltas_W1 = (alpha * gradients_W1) + (beta * deltas_W1)
    deltas_B1 = (alpha * gradients_B1) + (beta * deltas_B1)
    deltas_W2 = (alpha * gradients_W2) + (beta * deltas_W2)
    deltas_B2 = (alpha * gradients_B2) + (beta * deltas_B2)
    deltas_W3 = (alpha * gradients_W3) + (beta * deltas_W3)
    deltas_B3 = (alpha * gradients_B3) + (beta * deltas_B3)

    # Update parameters
    W1 -= (deltas_W1)
    B1 -= (deltas_B1)
    W2 -= (deltas_W2)
    B2 -= (deltas_B2)
    W3 -= (deltas_W3)
    B3 -= (deltas_B3)

    # Compute loss
    current_loss = loss(x, y, W1, B1, W2, B2, W3, B3)

    # Store MSE
    losses.append(current_loss)
    
    # Report?
    if((i % report_interval) == 0):
        np.set_printoptions(precision=3)
        print("MSE: {0:.2f}".format(current_loss))
end_time = time.time()

# Benchmark
elapsed = end_time - start_time
print("Time per iteration: {0} ms".format(1000*elapsed/num_epochs))

# Compare
prediction = func(x, W1, B1, W2, B2, W3, B3)
plt.plot(x, y, 'b.', markersize=1)
plt.plot(x, prediction, 'r.', markersize=1)
plt.show()

#plt.plot(np.array(losses))
#plt.show()

#FIN
