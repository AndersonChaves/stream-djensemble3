import matplotlib.pyplot as plt
import numpy as np

# Data
strategies = ['Best of all Models', 'Random Allocation', 'Average Ensemble', 'General Model', 'Stream Ensemble']
average_errors = [1.184, 14.447, 11.339, 21.482, 1.186]
std_deviations = [0.335, 15.131, 0.508, 0.492, 0.339]

# Create positions for bars
x = np.arange(len(strategies))

# Plot bars with error bars
plt.bar(x, average_errors, yerr=std_deviations, capsize=5)

# Title and labels
plt.title('Average Error of Different Strategies with Standard Deviation')
plt.xlabel('Strategies')
plt.ylabel('Average Error')

# Set x-axis ticks and labels
plt.xticks(x, strategies, rotation=45, ha='right')

# Show plot
plt.tight_layout()
plt.show()
