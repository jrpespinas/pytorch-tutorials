import math
import numpy as np

# Create random input and output data
x = np.linspace(-math.pi, math.pi, 2000)
y = np.sin(x)

# Randomly initialize weights
a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()

learning_rate = 1e-6
for t in range(2000):
    # Forward propagation
    # y = a + bx + cx^2 + dx^3
    # Horner's method form
    y_hat = a + x * (b + x * (c + x * d))

    loss = np.square(y_hat - y).sum()
    if t % 100 == 99:
        print(t, loss)


    # Backward propagation
    grad_y_hat = 2.0 * (y_hat - y)
    grad_a = grad_y_hat.sum()
    grad_b = (grad_y_hat * x).sum()
    grad_c = (grad_y_hat * x ** 2).sum()
    grad_d = (grad_y_hat * x ** 3).sum()

    # Update weights
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d


print(f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3')

