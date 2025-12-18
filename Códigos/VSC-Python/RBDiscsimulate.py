
# -*- coding: utf-8 -*-
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
import pandas as pd

# Define o modelo e suas dependências (arestas)
model = DiscreteBayesianNetwork([
    ("HISTORY", "HISTORY*"),
    ("HYPOVOLEMIA", "HISTORY*"),
    ("LVEDVOLUME", "HISTORY*")
])

# CPD para HISTORY
cpd_history = TabularCPD(
    variable="HISTORY",
    variable_card=2,
    values=[[0.5], [0.5]],
    state_names={"HISTORY": ["TRUE", "FALSE"]}
)

# CPD para HYPOVOLEMIA
cpd_hypovolemia = TabularCPD(
    variable="HYPOVOLEMIA",
    variable_card=2,
    values=[[0.5], [0.5]],
    state_names={"HYPOVOLEMIA": ["TRUE", "FALSE"]}
)

# CPD para LVEDVOLUME
cpd_lvedvolume = TabularCPD(
    variable="LVEDVOLUME",
    variable_card=3,
    values=[[0.3], [0.4], [0.3]],
    state_names={"LVEDVOLUME": ["LOW", "NORMAL", "HIGH"]}
)

# CPD para HISTORY*
cpd_history_star = TabularCPD(
    variable="HISTORY*",
    variable_card=2,
    values=[
        [0.2, 0.1, 0.6, 0.4, 0.7, 0.2, 0.5, 0.5, 0.4, 0.6, 0.7, 0.3],
        [0.8, 0.9, 0.4, 0.6, 0.3, 0.8, 0.5, 0.5, 0.6, 0.4, 0.3, 0.7],
    ],
    evidence=["HISTORY", "HYPOVOLEMIA", "LVEDVOLUME"],
    evidence_card=[2, 2, 3],
    state_names={
        "HISTORY*": [0, 1],
        "HISTORY": ["TRUE", "FALSE"],
        "HYPOVOLEMIA": ["TRUE", "FALSE"],
        "LVEDVOLUME": ["LOW", "NORMAL", "HIGH"]
    }
)

# Adiciona os CPDs
model.add_cpds(cpd_history, cpd_hypovolemia, cpd_lvedvolume, cpd_history_star)

# Verifica consistência
print("Modelo válido:", model.check_model())

# Simula 10 amostras (sem missing_prob)
samples = model.simulate(n_samples=10)

print("\nAmostras geradas:")
print(samples)

