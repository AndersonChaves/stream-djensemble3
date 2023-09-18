import matplotlib.pyplot as plt
from datetime import datetime

# Example data
timestamps = [datetime(2023, 7, 1, 10, 0), datetime(2023, 7, 1, 11, 0), datetime(2023, 7, 1, 12, 0)]
values = [10, 20, 15]

# Create the plot
plt.plot(timestamps, values, marker='o', linestyle='-')

# Add labels and title
plt.xlabel('Timestamp')
plt.ylabel('Value')
plt.title('Time Series Plot')

# Display the plot in a window
plt.show()
