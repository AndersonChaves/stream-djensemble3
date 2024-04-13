import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Data
strategies = ['Best of all Models', 'Random Allocation', 'Average Ensemble', 'General Model', 'Stream Ensemble (Stream QTree)']
average_errors = [1.184, 14.447, 11.339, 21.482, 1.186]
std_deviations = [0.335, 15.131, 0.508, 0.492, 0.339]

# Create DataFrame
data = {'Strategy': strategies, 'Average Error': average_errors, 'Standard Deviation': std_deviations}
df = pd.DataFrame(data)

# Set the style
sns.set(style="whitegrid")

# Plot with Seaborn
plt.figure(figsize=(8, 5))
sns.barplot(x='Strategy', y='Average Error', data=df, palette='muted', ci='sd')  # ci='sd' for error bars with standard deviation

# Add error bars
for i in range(len(strategies)):
    plt.errorbar(i, average_errors[i], yerr=std_deviations[i], fmt='none', ecolor='black', capsize=5, linewidth=1.5)

# Title and labels
plt.title('Average Error of Different Strategies with Standard Deviation', fontsize=16)
plt.xlabel('Strategies', fontsize=14)
plt.ylabel('Average Error', fontsize=14)

# Increase tick label font size
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Adjust layout to prevent clipping of labels
plt.tight_layout()

# Save the plot as a PDF file
plt.savefig('academic_plot.pdf')

# Show plot
plt.show()
