import numpy as np
import matplotlib.pyplot as plt

# generate Gaussian distributed data
data = np.random.normal(0, 1, 1000)

# plot the probability density function
plt.hist(data, density=True, bins=30)
plt.show()

# calculate the probability of obtaining a value greater than 1 standard deviation from the mean
p_value = 1 - np.cumsum(norm.cdf(1))