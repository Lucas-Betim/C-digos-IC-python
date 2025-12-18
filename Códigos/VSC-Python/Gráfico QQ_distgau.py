import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Generate some random data following a normal distribution
data = np.random.normal(0, 1, 1000)

# Create a QQ plot
sm.qqplot(data, line='s')

# Set plot title
plt.title('QQ Plot')

# Display the plot
plt.show()