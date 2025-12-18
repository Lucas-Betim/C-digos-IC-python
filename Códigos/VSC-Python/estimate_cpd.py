import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.factors.discrete import TabularCPD

# --- 1. DEFINIÇÃO DE DADOS E ESTRUTURA ---
# Dados fornecidos para a estimativa
data = pd.DataFrame(data={"A": [0, 0, 1], "B": [0, 1, 0], "C": [1, 1, 0]})
print("Dados de entrada:\n", data)

# Estrutura da Rede Bayesiana: A -> C, B -> C
model = DiscreteBayesianNetwork([("A", "C"), ("B", "C")])
print("\nEstrutura do modelo definida: A e B são pais de C.")

# --- 2. APRENDIZAGEM DE PARÂMETROS (CPDs) ---
# Inicializa o Estimador de Máxima Verossimilhança com o modelo e os dados
estimator = MaximumLikelihoodEstimator(model, data)

# Estima a CPD para o nó 'A' (Nó Raiz)
cpd_A = estimator.estimate_cpd(node="A")

# Estima a CPD para o nó 'B' (Nó Raiz)
cpd_B = estimator.estimate_cpd(node="B")

# Estima a CPD para o nó 'C' (Nó Filho de A e B)
# Note que, por padrão, o pgmpy pode aplicar suavização (smoothing) para combinações não observadas (como A=1, B=1).
cpd_C = estimator.estimate_cpd("C")

# --- 3. IMPRESSÃO DOS RESULTADOS ESTIMADOS ---
print("\n--- CPD Estimada para A (P(A)) ---")
print(cpd_A)

print("\n--- CPD Estimada para B (P(B)) ---")
print(cpd_B)

print("\n--- CPD Estimada para C (P(C | A, B)) ---")
print(cpd_C)