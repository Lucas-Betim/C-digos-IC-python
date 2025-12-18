# Importa as classes necessárias
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD

# 1. Cria a estrutura do modelo
# Define a rede bayesiana com as arestas A -> B e B -> C
model = DiscreteBayesianNetwork([("A", "B"), ("B", "C")])

# 2. Cria as Tabelas de Probabilidade Condicional (CPDs)

# CPD para A (nó sem pais)
# A tem 2 estados. P(A=0) = 0.2, P(A=1) = 0.8
cpd_a = TabularCPD("A", 2, [[0.2], [0.8]])

# CPD para B (depende de A)
# B tem 2 estados. A evidência é A, que tem 2 estados (cardinalidade 2).
# A primeira coluna é P(B|A=0), a segunda é P(B|A=1).
cpd_b = TabularCPD(
    "B", 2, [[0.3, 0.7], [0.7, 0.3]], evidence=["A"], evidence_card=[2]
)

# CPD para C (depende de B)
# C tem 2 estados. A evidência é B, que tem 2 estados (cardinalidade 2).
# A primeira coluna é P(C|B=0), a segunda é P(C|B=1).
cpd_c = TabularCPD(
    "C", 2, [[0.1, 0.9], [0.9, 0.1]], evidence=["B"], evidence_card=[2]
)

# 3. Adiciona as CPDs ao modelo principal
model.add_cpds(cpd_a, cpd_b, cpd_c)

# 4. Copia o modelo
copy_model = model.copy()

# 5. Imprime os resultados desejados

print("Modelo criado com sucesso!")

# Verifica os nós do modelo copiado (NodeView(('A', 'B', 'C')))
print("\nNós no modelo copiado:")
print(copy_model.nodes())

# Verifica as arestas do modelo copiado (OutEdgeView([('A', 'B'), ('B', 'C')]))
print("\nArestas no modelo copiado:")
print(copy_model.edges())

# Verifica o número de CPDs (tabelas de probabilidade) no modelo copiado (3)
print("\nNúmero de CPDs (tabelas de probabilidade) no modelo copiado:")
print(len(copy_model.get_cpds()))

# Opcional: Imprime uma das tabelas de probabilidade para confirmação
print("\nExemplo de CPD (para variável B):")
print(copy_model.get_cpds('B'))

print("\nExemplo de CPD (para variável C):")
print(copy_model.get_cpds('C'))