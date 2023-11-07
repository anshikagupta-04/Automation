
# import numpy as np
# from sklearn.linear_model import LinearRegression

# # Sample data (x, y coordinates)
# x = np.array([1, 2, 3, 4, 5, 6])
# y = np.array([2, 4, 5, 4, 6, 8])

# # Reshape x to a 2D array
# x = x.reshape(-1, 1)

# # Create a linear regression model
# model = LinearRegression()

# # Fit the model to the data
# model.fit(x, y)

# # Predict the next point
# next_x = 7
# next_x = np.array(next_x).reshape(-1, 1)
# next_y = model.predict(next_x)

# print(f"Predicted next point: ({next_x[0][0]}, {next_y[0]:.2f})")

# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.linear_model import LinearRegression
# from matplotlib import style
# import matplotlib.animation as animation


# # Generate random x values as timestamps
# x = np.sort(np.random.uniform(0, 10, 60))

# # Generate y values using a quadratic equation with some random noise
# y = 2*x**2 + 3*x + np.random.uniform(-5, 5, 60)

# xs=np.array(x, dtype=np.float64)
# ys=np.array(y, dtype=np.float64)

# plt.plot(x, y, color='blue')
# plt.xlabel('x', color='white')
# plt.ylabel('y', color='white')
# plt.show()

# xs = xs.reshape(-1, 1)

# # Create a linear regression model
# model = LinearRegression()

# # Fit the model to the data
# model.fit(xs, ys)


# # Predict the next point
# next_x = 10
# next_x = np.array(next_x).reshape(-1,1)
# next_y = model.predict(next_x)



# for nx in (1,6):
#     nx = np.sort(np.random.uniform(10, 20, 6))
#     nx = np.array(nx).reshape(-1,1)
#     ny = model.predict(nx)

# print(f"Predicted next point: ({next_x[0][0]}, {next_y[0]:.2f})")

# plt.scatter(next_x, next_y, color='red', s=75)
# plt.xlabel('x', color='white')
# plt.ylabel('y', color='white')
# a=plt.show()
# a

# print(f"Predicted next point: ({nx[0][0]}, {ny[0]:.2f})")
# plt.scatter(nx, ny, color='red')
# plt.xlabel('x', color='white')
# plt.ylabel('y', color='white')
# b=plt.show()
# b

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import style
import matplotlib.animation as animation


# Generate random x values as timestamps
x = np.sort(np.random.uniform(0, 10, 60))

# Generate y values using a quadratic equation with some random noise
y = 2*x**2 + 3*x + np.random.uniform(-5, 5, 60)

xs=np.array(x, dtype=np.float64)
ys=np.array(y, dtype=np.float64)

# plt.plot(x, y, color='blue')
# plt.xlabel('x', color='white')
# plt.ylabel('y', color='white')
# plt.show()

xs = xs.reshape(-1, 1)

# Create a linear regression model
model = LinearRegression()

# Fit the model to the data
model.fit(xs, ys)


# Predict the next point
next_x = 10
next_x = np.array(next_x).reshape(-1,1)
next_y = model.predict(next_x)

new_xs = []
new_ys = []
new_xs.extend(xs.tolist())
new_ys.extend(ys.tolist())

for nx in range(1,5):
    nx = np.sort(np.random.uniform(10, 15, 6))
    nx = np.array(nx).reshape(-1,1)
    ny = model.predict(nx)
    print(f"Predicted next point: ({nx[0][0]}, {ny[0]:.2f})")
    
new_xs.extend(nx.tolist())
new_ys.extend(ny.tolist())


plt.plot(xs,ys,color='blue')
plt.scatter(xs, ys, color='red', s=75)
plt.plot(nx,ny,color='blue')
plt.xlabel('x', color='white')
plt.ylabel('y', color='white')
a=plt.show()
a

# print(f"Predicted next point: ({nx[0][0]}, {ny[0]:.2f})")
# plt.scatter(nx, ny, color='red')
# plt.xlabel('x', color='white')
# plt.ylabel('y', color='white')
# b=plt.show()
# b