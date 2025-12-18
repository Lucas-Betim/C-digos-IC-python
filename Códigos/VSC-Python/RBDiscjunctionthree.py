from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
# A importação de JunctionTree é útil para fins de tipagem, mas o método retorna o objeto correto.
from pgmpy.estimators import ParameterEstimator 

# 1. Define a estrutura da Rede Bayesiana (Grafo G)
G = DiscreteBayesianNetwork(
    [
        ("diff", "grade"),
        ("intel", "grade"),
        ("intel", "SAT"),
        ("grade", "letter"),
    ]
)

# 2. Define as Tabelas de Probabilidade Condicional (CPDs)
diff_cpd = TabularCPD("diff", 2, [[0.2], [0.8]])
intel_cpd = TabularCPD("intel", 3, [[0.5], [0.3], [0.2]])
grade_cpd = TabularCPD(
    "grade",
    3,
    [
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
    ],
    evidence=["diff", "intel"],
    evidence_card=[2, 3],
)
sat_cpd = TabularCPD(
    "SAT",
    2,
    [[0.1, 0.2, 0.7], [0.9, 0.8, 0.3]],
    evidence=["intel"],
    evidence_card=[3],
)
letter_cpd = TabularCPD(
    "letter",
    2,
    [[0.1, 0.4, 0.8], [0.9, 0.6, 0.2]],
    evidence=["grade"],
    evidence_card=[3],
)

# 3. Adiciona as CPDs ao modelo e verifica sua validade
G.add_cpds(diff_cpd, intel_cpd, grade_cpd, sat_cpd, letter_cpd)
if not G.check_model():
    print("ERRO: O modelo não é consistente.")
    exit()

# 4. Converte a Rede Bayesiana em uma Árvore de Junção (Junction Tree)
jt = G.to_junction_tree()

# 5. Imprime a estrutura da Árvore de Junção
print("--- Conversão para Árvore de Junção ---")
print("Tipo do objeto retornado:", type(jt))
print("\nCliques (Nós da Árvore de Junção):")
print(jt.nodes)

print("\nConexões (Arestas da Árvore de Junção):")
print(jt.edges)