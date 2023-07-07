import plotly.graph_objects as go
import numpy as np
import random
from sympy import *
import matplotlib.pyplot as plt


# String to sympy
# vectors
# add momentum
# just the code no plotting
# support more variables

x, y, z = symbols('x y z', real=True)

f = "x**2+y**2-x*2"

#defining the function
def compute(x, y):
    return x**2+y**2-x*2

# give output coordinate when given input
def gradient(func, a, b):

    gx = diff(func, x)
    cx = gx.subs({x:a, y:b})

    gy = diff(func,y)
    cy = gy.subs({x:a, y: b})

    gradient=[cx, cy]

    return gradient

# pick a random point
xr = random.randint(-100, 100)
yr = random.randint(-100, 100)

# plot starting (random) point
print("Before: "+str(compute(xr, yr)))

# set learning rate
learning_rate = 0.01

def plot_loss():
   plt.plot(range(len(losses)), losses)
   plt.show()

#gradient descent implementation
def gradient_descent(x1, y1, learning_rate):

    losses = []
    xts = []
    yts = []

    for i in range(0,200):
        G = gradient(x**2+y**2-x*2, x1, y1)
        xt = x1 - G[0]*learning_rate
        yt = y1 - G[1]*learning_rate
        loss = (x**2+y**2-x*2).subs({x:xt, y:yt})
        losses.append(loss)
        xts.append(xt)
        yts.append(yt)
        x1 = xt
        y1 = yt

    return xt, yt, losses, xts, yts


xf, yf, losses, xts, yts = gradient_descent(xr, yr, learning_rate)

final_loss = compute(xf, yf)

print("After: " + str(final_loss))


# Generate data for the function
x2 = np.linspace(-100, 100, 100)
y2 = np.linspace(-100, 100, 100)
X, Y = np.meshgrid(x2, y2)
Z = compute(X, Y)


# Plot the function
fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z)])

fig.add_trace(go.Scatter3d(x=[xr], y=[yr], z=[compute(xr, yr)], mode='markers', marker=dict(size=5, color='blue')))

fig.add_trace(go.Scatter3d(x=[int(xf)], y=[int(yf)], z=[int(final_loss)], mode='markers', marker=dict(size=5, color='red')))

fig.show()

plot_loss()
