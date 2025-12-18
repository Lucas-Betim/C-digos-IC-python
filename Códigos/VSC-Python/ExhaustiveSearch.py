import pandas as pd
import numpy as np
# Importa o score de estrutura e o estimador de busca
from pgmpy.estimators import BIC, ExhaustiveSearch
# Importa a classe de modelo correta
from pgmpy.models import DiscreteBayesianNetwork

# 1. Criação de Dados (Z depende de X e Y)
data = pd.DataFrame(np.random.randint(
    0, 4, size=(5000, 2)), columns=list('XY'))
data['Z'] = data['X'] + data['Y']

# 2. Inicialização da Métrica de Score (BIC)
# O BICScore é utilizado para avaliar a qualidade de ajuste de cada estrutura.
bic = BIC(data)

# 3. Inicialização e Execução da Busca Exaustiva
# A Busca Exaustiva testa TODAS as ordens possíveis de edges entre os nós.
es = ExhaustiveSearch(data, scoring_method=bic)

# 4. Estimar o Melhor Modelo
# O modelo com o score BIC mais alto (menos negativo)
print("--- Resultado da Busca Exaustiva ---")
best_model = es.estimate()
print("\nModelo com o MAIOR score (BEST MODEL):")
print(best_model.edges())

# 5. Listar Todos os DAGs por Score
print("\nTodos os DAGs por Score (do maior para o menor):")
# O método all_scores() retorna (score, modelo_dag)
for score, dag in reversed(es.all_scores()):
    # Convertemos o modelo DAG de volta para o tipo correto para imprimir as edges
    model_instance = DiscreteBayesianNetwork(dag.edges())
    print(f"Score: {score:.4f} | Edges: {model_instance.edges()}")

print("-" * 40)
