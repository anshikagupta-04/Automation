import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from matplotlib.animation import FuncAnimation

# Generate random x values as timestamps
x = np.sort(np.random.uniform(0, 10, 60))

# Generate y values using a linear equation with some random noise
y = 2 * x  + np.random.uniform(-3, 3, 60)

# Create a linear regression model
model = LinearRegression()

# Reshape x for model fitting
x = x.reshape(-1, 1)

# Fit the model to the data
model.fit(x, y)

# Predict the next points
next_x = np.arange(10, 15, 1).reshape(-1, 1)
next_y = model.predict(next_x)

print(f"Predicted next point: ({next_x[0][0]}, {next_y[0]:.2f})")

# Initialize the plot
plt.scatter(x, y, color='blue', label='Data Points')
plt.plot(x, model.predict(x), color='red', label='Linear Regression')
plt.scatter(next_x, next_y, color='green', label='Predicted Points')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Linear Regression Prediction')

plt.scatter(x, y, label='Data Points')
plt.scatter(next_x, next_y, color='green')
plt.plot(next_x, next_y, color='red', label='Quadratic Curve')

# Animation
fig, ax = plt.subplots()
ax.set_xlim(0, 20) 
ax.set_ylim(0, 30)

ax.scatter(x, y, color='red', label='Data Points')

x_plot = next_x
y_plot = next_y
line, = ax.plot([], [], lw=2)

def animate(frame):
    if frame < len(x_plot):
        line.set_data(x_plot[:frame], y_plot[:frame])
        return line,

num_frames = len(x_plot)
anim = FuncAnimation(fig, animate, frames=num_frames, repeat=True, blit=True)
plt.show()

anim.save("predicted_points_animation.gif", writer="pillow") 


