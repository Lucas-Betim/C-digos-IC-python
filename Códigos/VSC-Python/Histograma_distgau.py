import matplotlib.pyplot as plt
import numpy as np

# Generate some random data following a normal distribution
data = np.random.normal(0, 1, 1000)

# Create a histogram plot
plt.hist(data, bins=30, edgecolor='black')

# Set plot labels and title
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Histogram Plot')

# Display the plot
plt.show()