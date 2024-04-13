import matplotlib.pyplot as plt
import numpy as np

# Error values
error_values = np.array([
    [2.4220, 2.6594, 2.6594, 2.6594, 2.6594, 2.6594],
    [2.0411, 2.0607, 1.8703, 2.0607, 2.0527, 1.8703],
    [1.1561, 1.1561, 1.1561, 1.1561, 1.1561, 1.1561],
    [1.2917, 1.2917, 1.2917, 1.2917, 1.2917, 1.2917],
    [1.7301, 1.7301, 1.7301, 1.7301, 1.7301, 1.7301],
    [2.1569, 2.0948, 2.0948, 2.0948, 2.0948, 2.0948],
    [1.7746, 1.7746, 1.7746, 1.7746, 1.7746, 1.7746],
    [1.9601, 1.9601, 1.9601, 1.9601, 1.9601, 1.9601],
    [2.6900, 2.9176, 2.9176, 2.9176, 2.6856, 2.6406],
    [0.7148, 0.7148, 0.7148, 0.7148, 0.7148, 0.7148]
])

# Standard deviation values
std_dev_values = np.array([
    [0.325, 0.000, 0.000, 0.000, 0.000, 0.000],
    [0.026, 0.000, 0.426, 0.000, 0.018, 0.426],
    [0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    [0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    [0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    [0.139, 0.000, 0.000, 0.000, 0.000, 0.000],
    [0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    [0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    [0.193, 0.000, 0.000, 0.000, 0.252, 0.619],
    [0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
])

# Labels for x-axis
labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']


# Plotting the error values with error bars representing standard deviation
plt.figure(figsize=(10, 6))
for i, metric in enumerate(['A', 'B', 'C', 'D', 'E', 'F']):
    plt.errorbar(labels, error_values[:, i], yerr=std_dev_values[:, i], label=metric, marker='o')

plt.title('Error Values with Standard Deviation')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.legend()
#plt.grid(True)
plt.show()
