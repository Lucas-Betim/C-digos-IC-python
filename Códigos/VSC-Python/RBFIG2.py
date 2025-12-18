from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import networkx as nx
import matplotlib.pyplot as plt

# 1. Definir a Estrutura da Rede (Nós e Arestas)pip install matplotlib networkx
modelo_family_out = DiscreteBayesianNetwork([
    ('family-out', 'light-on'),
    ('family-out', 'dog-out'),
    ('bowel-problem', 'dog-out'),
    ('dog-out', 'hear-bark')
])

# 2. Definir as Tabelas de Probabilidade Condicional (CPDs)

# CPD para 'family-out' (fo) - Nó Raiz
cpd_fo = TabularCPD(
    variable='family-out',
    variable_card=2,
    values=[[0.15],
            [0.85]]
)

# CPD para 'bowel-problem' (bp) - Nó Raiz
cpd_bp = TabularCPD(
    variable='bowel-problem',
    variable_card=2,
    values=[[0.01],
            [0.99]]
)

# CPD para 'light-on' (lo) - Depende de 'family-out' (fo)
cpd_lo = TabularCPD(
    variable='light-on',
    variable_card=2,
    values=[[0.60, 0.05],
            [0.40, 0.95]],
    evidence=['family-out'],
    evidence_card=[2]
)

# CPD para 'dog-out' (do) - Depende de 'family-out' (fo) e 'bowel-problem' (bp)
cpd_do = TabularCPD(
    variable='dog-out',
    variable_card=2,
    values=[[0.99, 0.97, 0.90, 0.30],
            [0.01, 0.03, 0.10, 0.70]],
    evidence=['family-out', 'bowel-problem'],
    evidence_card=[2, 2]
)

# CPD para 'hear-bark' (hb) - Depende de 'dog-out' (do)
cpd_hb = TabularCPD(
    variable='hear-bark',
    variable_card=2,
    values=[[0.70, 0.01],
            [0.30, 0.99]],
    evidence=['dog-out'],
    evidence_card=[2]
)

# 3. Anexar as CPDs ao Modelo
modelo_family_out.add_cpds(cpd_fo, cpd_bp, cpd_lo, cpd_do, cpd_hb)

# 4. Verificar se o Modelo é Válido
print("A Rede Bayesiana foi criada com sucesso.")
print(f"Modelo Válido: {modelo_family_out.check_model()}")

print("\n--- Tabela de Probabilidade Condicional (CPD) para 'family-out' ---")
print(cpd_fo)

print("\n--- Tabela de Probabilidade Condicional (CPD) para 'bowel-problem' ---")
print(cpd_lo)

print("\n--- Tabela de Probabilidade Condicional (CPD) para 'light-on' ---")
print(cpd_bp)

print("\n--- Tabela de Probabilidade Condicional (CPD) para 'dog-out' ---")
print(cpd_do)

print("\n--- Tabela de Probabilidade Condicional (CPD) para 'hear-bark' ---")
print(cpd_hb)

# 5. Desenhar o Grafo
print("\n--- Desenhando o Grafo da Rede Bayesiana ---")

plt.figure(figsize=(10, 8))

# Usando shell_layout, que é mais simples
pos = nx.shell_layout(modelo_family_out)

nx.draw(
    modelo_family_out,
    pos=pos,
    with_labels=True,
    node_size=3000,
    node_color="lightblue",
    font_size=10,
    arrowsize=10,        # Reduzido para 10
    edge_color="gray"
)

plt.title("Rede Bayesiana para o Problema 'family-out'")
plt.show()

# 4. (Continuação) Configuração para Inferência
inferencia = VariableElimination(modelo_family_out)

# --- 6. INFERÊNCIAS DE TESTE ---

# 1. Inferência a Priori: P(dog-out)
# (Qual a probabilidade de o cachorro estar fora sem nenhuma observação?)
prob_dog_out_apriori = inferencia.query(
    variables=['dog-out']
)
print("\n--- 1. priori P(dog-out) ---")
print(prob_dog_out_apriori)


# 2. Inferência Diagnóstica: P(family-out | hear-bark=True)
# (Se eu ouço o latido, qual a chance de a família ter saído?)
# No pgmpy: 0 = False, 1 = True (para variáveis binárias)
prob_fo_dado_hb = inferencia.query(
    variables=['family-out'],
    evidence={'hear-bark': 1}
)
print("\n--- 2. diagnostica P(family-out | hear-bark=True) ---")
print(prob_fo_dado_hb)


# 3. Inferência Intercausal (Explicativa): P(family-out | dog-out=True, bowel-problem=False)
# (Sabendo que o cachorro está fora, mas NÃO é por problema intestinal, qual a probabilidade de ser a família?)
prob_fo_intercausal = inferencia.query(
    variables=['family-out'],
    evidence={'dog-out': 1, 'bowel-problem': 0}
)
print("\n--- 3. intercasual P(family-out | dog-out=True, bowel-problem=False) ---")
print(prob_fo_intercausal)


# 4. Inferência Preditiva: P(hear-bark | family-out=True)
# (Se a família saiu, qual a chance de eu ouvir o latido?)
prob_hb_dado_fo = inferencia.query(
    variables=['hear-bark'],
    evidence={'family-out': 1}
)
print("\n--- 4. preditiva P(hear-bark | family-out=True) ---")
print(prob_hb_dado_fo)
