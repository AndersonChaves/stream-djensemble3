import matplotlib.pyplot as plt

# Data
strategies = ['Best of all Models', 'Random Allocation', 'Average Ensemble', 'General Model', 'Stream Ensemble (Stream QTree)']
average_errors = [1.184, 14.447, 11.339, 21.482, 1.186]
std_deviations = [0.335, 15.131, 0.508, 0.492, 0.339]

std_deviations[1] = 1

# Plot points with error bars
plt.errorbar(strategies, average_errors, yerr=std_deviations, fmt='o', markersize=8, capsize=5)

# Title and labels
plt.title('Average Error of Different Strategies with Standard Deviation')
plt.xlabel('Strategies')
plt.ylabel('Average Error')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Show plot
plt.tight_layout()
plt.show()

