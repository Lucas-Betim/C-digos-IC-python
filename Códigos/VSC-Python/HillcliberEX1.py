import pandas as pd
import numpy as np
# Importa o score e o estimador de busca
from pgmpy.estimators import BIC, HillClimbSearch 
from pgmpy.models import DiscreteBayesianNetwork 

# 1. Criação de Dados
# 8 variáveis (A-H), 2500 amostras
data = pd.DataFrame(np.random.randint(0, 3, size=(2500, 8)), columns=list('ABCDEFGH'))

# Adicionar dependências fortes nos dados:
# A depende de B e C
data['A'] = data['B'] + data['C'] 
# H depende de G e A (e, por extensão, de B e C)
data['H'] = data['G'] - data['A']
# Variáveis X, Y, Z são discretas (0, 1, 2) e agora A e H têm mais estados.
# Para evitar warnings, garantimos que todas as colunas sejam tratadas como categóricas:
for col in data.columns:
    data[col] = data[col].astype('category')


# 2. Inicialização da Métrica de Score (BIC)
# O BicScore é usado para avaliar a qualidade de cada estrutura testada.
bic = BIC(data)

# 3. Inicialização e Execução da Busca Hill Climb
# A busca tentará encontrar o modelo que maximiza o BicScore.
hc = HillClimbSearch(data)

# 'estimate' executa a busca e retorna a estrutura com o melhor score local
best_model = hc.estimate(scoring_method=bic)

# 4. Imprimir a Estrutura Encontrada
print("--- Estrutura Encontrada (Hill Climb Search) ---")
print("Edges do Melhor Modelo Encontrado:")
print(best_model.edges())