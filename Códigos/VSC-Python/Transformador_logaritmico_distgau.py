import numpy as np

# Generate some random skewed data
data = np.random.exponential(scale=1, size=1000)

# Apply logarithmic transformation
log_data = np.log(data)

# Plot original and transformed data
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Original data
axes[0].hist(data, bins=30)
axes[0].set_title('Original Data')

# Transformed data
axes[1].hist(log_data, bins=30)
axes[1].set_title('Log-Transformed Data')

plt.tight_layout()
plt.show()