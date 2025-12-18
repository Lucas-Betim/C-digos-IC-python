import numpy as np
import pandas as pd
# Corrigido: Substituído BayesianModel por DiscreteBayesianNetwork
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator

# --- 1. Gerar Dados Aleatórios ---
# 5000 amostras com 4 colunas binárias (0 ou 1)
# Seus dados são aleatórios, o que significa que as variáveis são independentes
# As probabilidades resultantes devem ser próximas de 0.5 para cada estado.
data = pd.DataFrame(np.random.randint(low=0, high=2, size=(5000, 4)), 
                    columns=['A', 'B', 'C', 'D'])

# --- 2. Definir a Estrutura do Modelo ---
# model = BayesianModel([('A', 'B'), ('A', 'C'), ('D', 'C'), ('B', 'D')])
# Usando a classe correta
model = DiscreteBayesianNetwork([
    ('A', 'B'), 
    ('A', 'C'), 
    ('D', 'C'), 
    ('B', 'D')
])

# --- 3. Ajustar o Modelo usando o Estimador Bayesiano ---
print("--- Ajustando o Modelo com BayesianEstimator (BDeu) ---")

# fit(data, estimator=BayesianEstimator, prior_type="BDeu")
model.fit(data, 
          estimator=BayesianEstimator, 
          prior_type="BDeu",
          # O default equivalent_sample_size é 1, mas você pode especificar
          equivalent_sample_size=1 
         )

print("\nCPDs Estimadas e Adicionadas ao Modelo:")

# for cpd in model.get_cpds():
#     print(cpd)
for cpd in model.get_cpds():
    print("---------------------------------")
    print(cpd)
print("---------------------------------")