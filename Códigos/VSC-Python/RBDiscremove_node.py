


import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork

# 1. Define a estrutura inicial do modelo
model = DiscreteBayesianNetwork(
    [("A", "B"), ("B", "C"), ("A", "D"), ("D", "C")]
)

# 2. Gera dados aleatórios e treina o modelo
values = pd.DataFrame(
    np.random.randint(low=0, high=2, size=(1000, 4)),
    columns=["A", "B", "C", "D"],
)
model.fit(values)

# --- INÍCIO DA CORREÇÃO ---
def print_cpd_details(model_instance):
    """Imprime a variável, seus pais e a tabela de probabilidade (values)."""
    
    cpds = model_instance.get_cpds()
    if not cpds:
        print("Nenhuma CPD presente no modelo.")
        return

    for cpd in cpds:
        parents = cpd.get_evidence()
        print(f"\n- Variável: {cpd.variable} | Pais: {parents}")
        print(f"  Cardinalidade: {cpd.cardinality}")
        
        # Imprime a Tabela de Probabilidade (os valores numéricos)
        print("  Tabela (Valores):")
        # Usamos o método to_numpy() para garantir que a saída seja limpa
        print(cpd.values) 
# --- FIM DA CORREÇÃO ---


print("--- 1. Estado do Modelo ANTES de remover 'A' ---")
print("Nós (Variáveis):", model.nodes())
print("Arestas (Conexões):", model.edges())
print("\nCPDs (Probabilidades) antes da remoção:")
print_cpd_details(model)


# 3. Remove o nó "A"
model.remove_node("A")


print("\n=======================================================")
print("--- 2. Estado do Modelo DEPOIS de remover 'A' ---")
print("Nós (Variáveis):", model.nodes())
print("Arestas (Conexões):", model.edges())
print("\nCPDs (Probabilidades) depois da remoção:")
print_cpd_details(model)