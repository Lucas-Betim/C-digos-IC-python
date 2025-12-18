'''
from scipy import stats
import matplotlib.pyplot as plt

# Generate some non-normal data
data = stats.loggamma.rvs(5, size=1000)

# Apply Box-Cox transformation
transformed_data, lambda_ = stats.boxcox(data)

# Plot original and transformed data
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Original data
axes[0].hist(data, bins=30)
axes[0].set_title('Original Data')

# Transformed data
axes[1].hist(transformed_data, bins=30)
axes[1].set_title('Box-Cox Transformed Data (λ={:.2f})'.format(lambda_))

plt.tight_layout()
plt.show()
'''

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# 1. Gerar dados não-normais.
# (Exemplo usando Log-Gamma, que é geralmente > 0)
data = stats.loggamma.rvs(5, size=1000)

# --- GARANTIA DE POSITIVIDADE PARA O BOX-COX ---
min_value = np.min(data)
positive_data = data

if min_value <= 0:
    # Se houver valores não-positivos, aplica um deslocamento.
    # O deslocamento 'c' garante que o novo valor mínimo seja 0.001.
    c = abs(min_value) + 0.001
    positive_data = data + c
    print(f"ATENÇÃO: Dados deslocados por {c:.4f} para garantir a positividade do Box-Cox.")
else:
    print("Dados já são positivos. Nenhum deslocamento foi necessário.")
# --- FIM DA GARANTIA ---

# 2. Aplicar a Transformação BOX-COX
transformed_data, lambda_ = stats.boxcox(positive_data)

# 3. Plotar os resultados
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Histograma dos Dados Originais (ou Deslocados)
axes[0].hist(positive_data, bins=30)
axes[0].set_title('Dados Originais (Ajustados)')
axes[0].set_xlabel('Valores')
axes[0].set_ylabel('Frequência')

# Histograma dos Dados Transformados
axes[1].hist(transformed_data, bins=30)
axes[1].set_title('Box-Cox Transformado (λ={:.2f})'.format(lambda_))
axes[1].set_xlabel('Valores Transformados')
axes[1].set_ylabel('Frequência')

plt.suptitle('Comparação de Distribuições (Usando Box-Cox)', fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()