import math
import torch

dtype = torch.float
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create random input and output data
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

# Randomly initialize weights
a = torch.randn((), device=device, dtype=dtype)
b = torch.randn((), device=device, dtype=dtype)
c = torch.randn((), device=device, dtype=dtype)
d = torch.randn((), device=device, dtype=dtype)

learning_rate = 1e-6
for t in range(2000):
    # Forward propagation
    # y = a + bx + cx^2 + dx^3
    y_hat = a + x * (b + x * (c + x * d))

    # Compute Loss
    loss = (y_hat - y).pow(2).sum().item()
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

print(f'y = {a} + {b}x + {c}x^2 + {d}x^3')
