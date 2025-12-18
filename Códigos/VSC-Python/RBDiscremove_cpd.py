from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD

# 1. Define a estrutura da Rede Bayesiana
student = DiscreteBayesianNetwork([("diff", "grade"), ("intel", "grade")])

# 2. Cria a Tabela de Probabilidade Condicional (CPD)
cpd = TabularCPD(
    "grade",
    2,
    [[0.1, 0.9, 0.2, 0.7], [0.9, 0.1, 0.8, 0.3]],
    evidence=["intel", "diff"],
    evidence_card=[2, 2],
)

# 3. Adiciona a CPD ao modelo
student.add_cpds(cpd)
print("--- Após adicionar a CPD ---")
print(f"CPDs no modelo: {student.get_cpds()}") # Deve listar 1 CPD

# 4. Remove a CPD do modelo
# Removemos a CPD usando o objeto CPD que criamos.
student.remove_cpds(cpd)
print("\n--- Após remover a CPD ---")
print(f"CPDs no modelo: {student.get_cpds()}") # Deve retornar uma lista vazia

# Alternativa: Remover usando o nome da variável
# student.remove_cpds("grade")