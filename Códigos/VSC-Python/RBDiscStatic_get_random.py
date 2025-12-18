from pgmpy.models import DiscreteBayesianNetwork

# 1. Gera uma Rede Bayesiana Aleatória
# n_nodes=5 cria um gráfico com 5 variáveis.
# Como nenhum outro parâmetro (edge_prob, n_states, seed) foi fornecido,
# ele usa os valores padrão para gerar a estrutura e as probabilidades.
model = DiscreteBayesianNetwork.get_random(n_nodes=5)

print("--- Modelo Aleatório Gerado ---")

# 2. Imprime os nós
# O get_random nomeia os nós como inteiros, mas a ordem pode ser aleatória.
print("Nós do modelo:")
print(model.nodes())

# 3. Imprime as arestas
# Mostra as dependências (pai -> filho) geradas aleatoriamente.
print("\nArestas (Estrutura) do modelo:")
print(model.edges())

# 4. Imprime as CPDs (Tabelas de Probabilidade Condicional)
# O modelo já vem com probabilidades aleatórias (CPDs) anexadas.
print("\nCPDs (Tabelas de Probabilidade) anexadas:")
print(model.cpds)
# A saída mostra a relação P(filho | pais) para cada nó gerado.



