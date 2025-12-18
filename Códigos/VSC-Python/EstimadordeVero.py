import numpy as np
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator

# --- 1. PREPARAÇÃO DOS DADOS ---
# Gerando um conjunto de dados aleatório (1000 amostras) para simular o treinamento
print("Gerando 1000 amostras de dados aleatórios...")
data = pd.DataFrame(
    np.random.randint(low=0, high=2, size=(1000, 5)),
    columns=["A", "B", "C", "D", "E"],
)
# print(data.head()) # Linhas de cabeçalho do DataFrame

# --- 2. DEFINIÇÃO DA ESTRUTURA DA REDE ---
# Definindo as dependências causais (arestas)
model = DiscreteBayesianNetwork(
    [("A", "B"), ("C", "B"), ("C", "D"), ("B", "E")]
)
print("Estrutura da Rede Bayesiana definida: A->B, C->B, C->D, B->E.")


# --- 3. APRENDIZAGEM DE PARÂMETROS ---
# Estima as Tabelas de Probabilidade Condicional (CPDs) 
# usando o Estimador de Máxima Verossimilhança (Maximum Likelihood Estimator - MLE)
# O método .fit() calcula as probabilidades a partir do DataFrame 'data' e as anexa ao 'model'
print("Iniciando a aprendizagem de parâmetros (CPDs) usando MLE...")
model.fit(data, estimator=MaximumLikelihoodEstimator)


# --- 4. VERIFICAÇÃO E IMPRESSÃO DOS RESULTADOS ---
print("\n--- Resultados da Aprendizagem ---")

# Verifica se o modelo está válido
print(f"Modelo Válido após fit(): {model.check_model()}")

# Imprime a CPD aprendida para o nó 'B' (que tem pais 'A' e 'C')
print("\nCPD Estimada para o nó 'B' (dependente de A e C):")
print(model.get_cpds('B'))

# Imprime a CPD aprendida para o nó 'A' (que é nó raiz)
print("\nCPD Estimada para o nó 'A' (Nó Raiz):")
print(model.get_cpds('A'))

# Imprime a CPD aprendida para o nó 'E' (dependente de 'B')
print("\nCPD Estimada para o nó 'E' (dependente de B):")
print(model.get_cpds('E'))