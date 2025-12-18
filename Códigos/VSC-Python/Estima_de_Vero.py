import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator

# 1. Definir o Modelo de Rede Bayesiana (Estrutura)
# O modelo implica que 'fruit' é independente e 'tasty' depende de 'fruit' e 'size'.
model = DiscreteBayesianNetwork([
    ('fruit', 'tasty'),
    ('size', 'tasty')
])

# 2. Criar um Conjunto de Dados de Exemplo
# Os dados são projetados para produzir as probabilidades que você indicou
# (ou probabilidades próximas, dado que as CPDs são estimadas a partir da contagem).
data = pd.DataFrame(data={
    'fruit': ['apple', 'apple', 'apple', 'apple', 'banana', 'banana', 'banana', 'banana'],
    'size':  ['large', 'large', 'large', 'large', 'small', 'small', 'small', 'small'],
    'tasty': ['yes', 'yes', 'yes', 'no', 'no', 'no', 'no', 'no']
})

# 3. Inicializar o Estimador de Máxima Verossimilhança (MLE)
mle = MaximumLikelihoodEstimator(model, data)

# 4. Estimar e Imprimir a CPD Incondicional para 'fruit'
# Nota: Como 'fruit' não tem pais no modelo, esta é uma CPD incondicional.
# Contagens: 'apple' = 4, 'banana' = 4. Total = 8. Prob: 4/8 = 0.5.
print("--- CPD para 'fruit' (incondicional) ---")
print(mle.estimate_cpd('fruit'))
print("-" * 40)

# 5. Estimar e Imprimir a CPD Condicional para 'tasty'
# 'tasty' é condicional a 'fruit' e 'size'.
#
# Da nossa 'data':
# - Se fruit=apple e size=large: ('yes', 'yes', 'yes', 'no'). P(yes)=3/4=0.75; P(no)=1/4=0.25.
# - Se fruit=banana e size=small: ('no', 'no', 'no', 'no'). P(yes)=0/4=0.0; P(no)=4/4=1.0.
print("--- CPD para 'tasty' (condicional) ---")
print(mle.estimate_cpd('tasty'))
print("-" * 40)