# 1. Importar as bibliotecas necessárias
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

import matplotlib.pyplot as plt
import networkx as nx

# 2. Definir a estrutura da rede
model = DiscreteBayesianNetwork([('A', 'C'), ('B', 'C')])

# 3. Criar as CPDs
cpd_a = TabularCPD(variable='A', variable_card=2,
                   values=[[0.1], [0.9]])

cpd_b = TabularCPD(variable='B', variable_card=2,
                   values=[[0.4], [0.6]])

cpd_c = TabularCPD(
    variable='C',
    variable_card=2,
    values=[
        [0.99, 0.9, 0.9, 0.0],  # P(C=0 | A,B)
        [0.01, 0.1, 0.1, 1.0]   # P(C=1 | A,B)
    ],
    evidence=['A', 'B'],
    evidence_card=[2, 2]
)

# 4. Adicionar CPDs ao modelo
model.add_cpds(cpd_a, cpd_b, cpd_c)

# 5. Verificar se o modelo é válido
assert model.check_model()

# 6. Visualizar o grafo corretamente
G = nx.DiGraph(model.edges())

pos = {
    'A': (0, 1),
    'B': (2, 1),
    'C': (1, 0)
}

nx.draw(G, pos, with_labels=True,
        node_size=2000, node_color='skyblue',
        arrowsize=25)
plt.title("Rede Bayesiana A → C ← B")
plt.axis("off")
plt.show()

# 7. Inferência
infer = VariableElimination(model)
resultado = infer.query(variables=['C'], evidence={'A': 1, 'B': 0})

print("Distribuição P(C | A=1, B=0):")
print(resultado)
