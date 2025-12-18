import numpy as np
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD

# 1. Preparação dos Dados
values = pd.DataFrame(
    np.random.randint(low=0, high=2, size=(100, 5)),
    columns=["A", "B", "C", "D", "E"],
)

# Dados de treino e previsão
train_data = values[:80]
predict_data = values[80:]

# 2. Criação e Treinamento do Modelo
model = DiscreteBayesianNetwork(
    [("A", "B"), ("C", "B"), ("C", "D"), ("B", "E")]
)

# Treinamento do modelo
model.fit(train_data)

# 3. Preparação da Evidência
predict_data = predict_data.copy()

# Remove a variável 'B' para que o modelo a preveja (Corrigido)
predict_data.drop("B", axis=1, inplace=True) 

# Remove a variável 'E' para que o modelo a preveja (Corrigido)
# O ERRO ESTAVA AQUI: As aspas foram consertadas.
predict_data.drop("E", axis=1, inplace=True) 

# 4. Cálculo da Distribuição de Probabilidade
y_prob = model.predict_probability(predict_data)

# 5. Impressão dos Resultados
print("\n--- Distribuição de Probabilidade Prevista para B ---")
print(y_prob.head()) # Usando .head() para manter a saída concisa