from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.factors.discrete import JointProbabilityDistribution

# 1. Cria a Estrutura da Rede Bayesiana (Grafo G)
G = DiscreteBayesianNetwork([("diff", "grade"), ("intel", "grade")])

# 2. Define as CPDs (Probabilidades Condicionais)
cpd_diff = TabularCPD("diff", 2, [[0.2], [0.8]])
cpd_intel = TabularCPD("intel", 3, [[0.5], [0.3], [0.2]])
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

# Adiciona as CPDs ao grafo G
G.add_cpds(cpd_diff, cpd_intel, grade_cpd)

# 3. Define a Distribuição de Probabilidade Conjunta (JPD)
val = [
    0.01, 0.01, 0.08, 0.006, 0.006, 0.048, 0.004, 0.004, 0.032,
    0.04, 0.04, 0.32, 0.024, 0.024, 0.192, 0.016, 0.016, 0.128,
]
JPD = JointProbabilityDistribution(
    ["diff", "intel", "grade"], [2, 3, 3], val
)

# 4. Verifica se o grafo G é um I-Mapa da JPD
is_imap_result = G.is_imap(JPD)

print("--- Verificação I-Mapa ---")
print(f"Estrutura da Rede Bayesiana (G) é um I-Mapa da JPD? {is_imap_result}")
