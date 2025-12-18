import numpy as np
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork 
from pgmpy.estimators import BayesianEstimator

# 1. Gerar Dados Aleatórios
# 5000 amostras, 4 colunas (A, B, C, D), valores discretos (0 ou 1)
data = pd.DataFrame(np.random.randint(low=0, high=2, size=(5000, 4)), columns=['A', 'B', 'C', 'D'])

# 2. Definir o Modelo de Rede Bayesiana (Estrutura)
# A -> B, A -> C, D -> C, B -> D
model = DiscreteBayesianNetwork([('A', 'B'), ('A', 'C'), ('D', 'C'), ('B', 'D')])

# 3. Estimar Parâmetros Usando BayesianEstimator (BDeu)
# A prior BDeu é usada para suavizar as estimativas de probabilidade.
# equivalent_sample_size=5 é o padrão, que é um 'peso' dado à prior.
model.fit(
    data, 
    estimator=BayesianEstimator, 
    prior_type="BDeu", 
    # equivalent_sample_size=5 (Padrão, não precisa ser especificado)
)

# 4. Imprimir as CPDs Estimadas
# O resultado exato da CPD muda a cada execução porque os dados são aleatórios.
print("--- CPDs Estimadas com Prior BDeu ---")
for cpd in model.get_cpds():
    print(cpd)
    print("-" * 30)