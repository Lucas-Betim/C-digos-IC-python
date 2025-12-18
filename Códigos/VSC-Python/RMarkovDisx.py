from pgmpy.models import DiscreteBayesianNetwork

# 1. Define a estrutura da Rede Bayesiana (Grafo G)
G = DiscreteBayesianNetwork(
    [
        ("diff", "grade"),
        ("intel", "grade"),
        ("intel", "SAT"),
        ("grade", "letter"),
    ]
)

# 2. Converte para o Modelo de Markov (Grafo Moral)
# O método to_markov_model() retorna a classe correta (MarkovModel/DiscreteMarkovNetwork)
mm = G.to_markov_model()

print("--- Conversão para Modelo de Markov ---")

# 3. Imprime os nós e arestas (o que confirma o sucesso)
print("Tipo do objeto retornado:", type(mm))
print("Nós do Modelo de Markov:", mm.nodes())
print("Arestas do Modelo de Markov:", mm.edges())