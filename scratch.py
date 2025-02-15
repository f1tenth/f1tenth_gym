import matplotlib.pyplot as plt
import numpy as np

data = [1, 2, 3, 4, 5]

def arrow(start, end):
    ax.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1], head_width=10, head_length=10, fc='black', ec='black')

def polar_arrow(origin, r, theta):
    x = origin[0] + r * np.cos(theta)
    y = origin[1] + r * np.sin(theta)
    arrow(origin, (x, y))

def point(x, y):
    ax.scatter(x, y, color='blue', s=100)

def foo(data, winding_dir: str='CCW', starting_angle=-np.pi/2, scaling_factor: float = 10):
    assert winding_dir in ['CW', 'CCW']

    # if CW, add. if CCW, subtract
    dir = 1 if winding_dir == 'CW' else -1

    center = (255/2, 255/2)
    n = len(data)
    c = np.pi * 0.5 / n
    dtheta = dir * n

    point(center[0], center[1])

    for i, r in enumerate(data):
        theta = starting_angle + 2 * np.pi * i / dtheta
        polar_arrow(center, scaling_factor * r, theta)


# Create a figure and axis
fig, ax = plt.subplots()

foo(data)

# Set axis limits
ax.set_xlim(0, 256)
ax.set_ylim(0, 256)
ax.invert_yaxis()

plt.grid(True)
plt.show()
