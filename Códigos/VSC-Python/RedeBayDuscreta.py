# RedeBayDiscreta.py

from pgmpy.models import DiscreteBayesianNetwork   
from pgmpy.factors.discrete.CPD import TabularCPD

student = DiscreteBayesianNetwork(
    [("diff", "grades"), ("aptitude", "grades")]
)

grades_cpd = TabularCPD(
    "grades",
    3,
    [
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
    ],
    evidence=["diff", "aptitude"],
    evidence_card=[2, 3],
    state_names={
        "grades": ["gradeA", "gradeB", "gradeC"],
        "diff": ["easy", "hard"],
        "aptitude": ["low", "medium", "high"],
    },
)

student.add_cpds(grades_cpd)

print("Modelo de Rede Bayesiana criado com sucesso!")
print("\nEstrutura (Arestas) do Modelo:")
print(student.edges)

print("\nTabela de Probabilidade Condicional (CPD) de 'grades':")
print(grades_cpd)
