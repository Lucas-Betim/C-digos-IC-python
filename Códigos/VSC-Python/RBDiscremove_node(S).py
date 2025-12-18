import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork

# 1. Define a estrutura inicial do modelo: A -> B, B -> C, A -> D, D -> C
model = DiscreteBayesianNetwork(
    [("A", "B"), ("B", "C"), ("A", "D"), ("D", "C")]
)

# 2. Gera dados aleatórios e treina o modelo para obter as CPDs
values = pd.DataFrame(
    np.random.randint(low=0, high=2, size=(1000, 4)),
    columns=["A", "B", "C", "D"],
)
model.fit(values)

# Função auxiliar para imprimir detalhes da CPD, incluindo os valores da Tabela
def print_cpd_details(model_instance):
    """Imprime a variável, seus pais e a tabela de probabilidade (os valores)."""
    
    cpds = model_instance.get_cpds()
    if not cpds:
        print("Nenhuma CPD presente no modelo.")
        return

    for cpd in cpds:
        parents = cpd.get_evidence()
        print(f"\n- Variável: {cpd.variable} | Pais: {parents}")
        print(f"  Cardinalidade: {cpd.cardinality}")
        
        # IMPRIME A TABELA DE VALORES
        print("  Tabela (Valores):")
        print(cpd.values) 


print("--- 1. Estado do Modelo ANTES de remover 'A' e 'B' ---")
print_cpd_details(model)


# 3. Executa a remoção de múltiplos nós
model.remove_nodes_from(["A", "B"])


print("\n=======================================================")
print("--- 2. Estado do Modelo DEPOIS de remover 'A' e 'B' ---")
# Agora, C e D são as únicas variáveis restantes
print_cpd_details(model)