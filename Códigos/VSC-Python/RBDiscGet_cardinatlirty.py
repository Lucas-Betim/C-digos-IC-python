from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from collections import defaultdict

# 1. Define a estrutura da Rede Bayesiana
# A nota ('grade') depende da dificuldade ('diff') e da inteligência ('intel')
student = DiscreteBayesianNetwork([("diff", "grade"), ("intel", "grade")])

# 2. Define a Tabela de Probabilidade Condicional (CPD) para a dificuldade ('diff')
# 'diff' tem 2 estados. P(diff=0) = 0.6, P(diff=1) = 0.4
cpd_diff = TabularCPD("diff", 2, [[0.6], [0.4]])

# 3. Define a CPD para a inteligência ('intel')
# 'intel' tem 2 estados. P(intel=0) = 0.7, P(intel=1) = 0.3
cpd_intel = TabularCPD("intel", 2, [[0.7], [0.3]])

# 4. Define a CPD para a nota ('grade')
# 'grade' tem 2 estados. Evidências: 'intel' (2 estados) e 'diff' (2 estados).
# O número total de colunas é 2 * 2 = 4.
cpd_grade = TabularCPD(
    "grade",
    2,
    [[0.1, 0.9, 0.2, 0.7], [0.9, 0.1, 0.8, 0.3]],
    evidence=["intel", "diff"],
    evidence_card=[2, 2],
)

# 5. Adiciona as CPDs ao modelo
student.add_cpds(cpd_diff, cpd_intel, cpd_grade)

# 6. Imprime as cardinalidades (o número de estados de cada variável)

# a) Cardinalidade de todas as variáveis
print("Cardinalidade de todas as variáveis:")
print(student.get_cardinality())
# Saída esperada: defaultdict(<class 'int'>, {'diff': 2, 'intel': 2, 'grade': 2})

# b) Cardinalidade de uma variável específica ('intel')
print("\nCardinalidade da variável 'intel':")
print(student.get_cardinality("intel"))
# Saída esperada: 2