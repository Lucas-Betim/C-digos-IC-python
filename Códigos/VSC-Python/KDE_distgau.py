'''
import seaborn as sns
import numpy as np

# Generate some random data following a normal distribution
data = np.random.normal(0, 1, 1000)

# Create a kernel density plot
sns.kdeplot(data, shade=True)

# Set plot labels and title
plt.xlabel('Values')
plt.ylabel('Density')
plt.title('Kernel Density Plot')

# Display the plot
plt.show()
'''
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt # <--- ESSENCIAL PARA CORRIGIR O 'plt'

# 1. Gerar dados (Distribuição Normal/Gaussiana)
data = np.random.normal(loc=0, scale=1, size=1000)

# 2. Criar o gráfico de Estimativa de Densidade de Kernel (KDE)
# Corrigindo o 'shade=True' para 'fill=True'
sns.kdeplot(data, fill=True) 

# 3. Adicionar rótulos e mostrar o gráfico
plt.title('KDE da Distribuição Gaussiana')
plt.xlabel('Valores')
plt.ylabel('Densidade')
plt.show()