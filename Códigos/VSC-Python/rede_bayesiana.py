import networkx as nx
import matplotlib.pyplot as plt

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# =========================
# 1. Estrutura da Rede
# =========================
model = DiscreteBayesianNetwork([
    ("A", "B"),
    ("B", "E"),
    ("B", "F"),
    ("C", "D"),
    ("D", "F"),
    ("E", "H"),
    ("E", "I"),
    ("F", "G"),
])

# =========================
# 2. CPDs 
# =========================
cpd_A = TabularCPD("A", 2, [[0.9], [0.1]])
cpd_C = TabularCPD("C", 2, [[0.5], [0.5]])

cpd_B = TabularCPD(
    "B", 2,
    [[0.1, 0.6],
     [0.9, 0.4]],
    evidence=["A"], evidence_card=[2]
)

cpd_D = TabularCPD(
    "D", 2,
    [[0.4, 0.7],
     [0.6, 0.3]],
    evidence=["C"], evidence_card=[2]
)

cpd_E = TabularCPD(
    "E", 2,
    [[0.8, 0.9],
     [0.2, 0.1]],
    evidence=["B"], evidence_card=[2]
)

cpd_H = TabularCPD(
    "H", 2,
    [[0.3, 0.5],
     [0.7, 0.5]],
    evidence=["E"], evidence_card=[2]
)

cpd_I = TabularCPD(
    "I", 2,
    [[0.3, 0.2],
     [0.7, 0.8]],
    evidence=["E"], evidence_card=[2]
)

cpd_F = TabularCPD(
    "F", 2,
    [[1.0, 0.8, 0.6, 0.3],
     [0.0, 0.2, 0.4, 0.7]],
    evidence=["B", "D"], evidence_card=[2, 2]
)

cpd_G = TabularCPD(
    "G", 2,
    [[0.9, 0.1],
     [0.1, 0.9]],
    evidence=["F"], evidence_card=[2]
)

model.add_cpds(
    cpd_A, cpd_C, cpd_B, cpd_D,
    cpd_E, cpd_F, cpd_G, cpd_H, cpd_I
)

assert model.check_model()

# =========================
# 3. Inferência (exemplo)
# =========================
infer = VariableElimination(model)
print("P(G | A=1, C=0):")
print(infer.query(variables=["G"], evidence={"A": 1, "C": 0}))

# =========================
# 4. DESENHO 
# =========================
G_vis = nx.DiGraph(model.edges())

# layout hierárquico (manual, sem nós colados)
pos = {
    "A": (0, 3),
    "B": (0, 2),
    "E": (-1, 1),
    "F": (1, 1),
    "H": (-2, 0),
    "I": (0, 0),
    "G": (2, 0),
    "C": (3, 3),
    "D": (3, 2),
}

plt.figure(figsize=(11, 7))
nx.draw(
    G_vis, pos,
    with_labels=True,
    node_size=2000,
    node_color="skyblue",
    arrowsize=25,
    width=2
)
plt.title("Estrutura da Rede Bayesiana (pgmpy)")
plt.axis("off")
plt.show()





    
