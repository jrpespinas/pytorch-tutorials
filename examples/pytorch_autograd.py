import math
import torch

dtype = torch.float
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

x = torch.linspace(-math.pi, math.pi, device=device, dtype=dtype)
y = torch.sin(x)

# `requires_grad=True` indicates we want gradient computation
# with respect to these Tensors during backward pass
a = torch.randn((), device=device, dtype=dtype, requires_grad=True)
b = torch.randn((), device=device, dtype=dtype, requires_grad=True)
c = torch.randn((), device=device, dtype=dtype, requires_grad=True)
d = torch.randn((), device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(2000):
    y_hat = a + x * (b + x * (c + x * d))

    loss = (y_hat - y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())

    loss.backward()

    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad

        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None

print(f'Result: y={a} + {b}x + {c}x ^ 2 + {d}x ^ 3')
