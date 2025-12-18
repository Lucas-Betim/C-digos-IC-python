from pgmpy.models import DiscreteBayesianNetwork
# A importação de TabularCPD não é estritamente necessária para este trecho,
# mas mantê-la é útil para o contexto geral do pgmpy.
from pgmpy.factors.discrete import TabularCPD

# 1. Define a estrutura da Rede Bayesiana (G)
# Lista de arestas que definem as dependências (pai -> filho)
G = DiscreteBayesianNetwork(
    [
        ("x", "y"),
        ("z", "y"),
        ("y", "w"),
        ("y", "v"),
        ("u", "w"),
        ("s", "v"),
        ("w", "t"),
        ("w", "m"),
        ("v", "n"),
        ("v", "q"),
    ]
)

# 2. Obtém o Manto de Markov do nó "y"
# O Manto de Markov (Markov Blanket) de um nó é o conjunto de seus:
# - Pais
# - Filhos
# - Outros pais de seus filhos (co-pais)
markov_blanket_y = G.get_markov_blanket("y")
markov_blanket_v = G.get_markov_blanket("v")

# 3. Imprime o resultado
print("O Manto de Markov do nó 'y' é:")
print(markov_blanket_y)
# Saída esperada (a ordem pode variar): ['s', 'u', 'w', 'v', 'z', 'x']
print("O Manto de Markov do nó 'v' é:")
print(markov_blanket_v)