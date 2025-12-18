import numpy as np
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
# Importa a classe de inferência aproximada necessária para o método .predict()
from pgmpy.inference import ApproxInference 

# 1. Preparação dos Dados
# Cria 1000 linhas de dados aleatórios (0 ou 1) para 5 variáveis (A a E)
values = pd.DataFrame(
    np.random.randint(low=0, high=2, size=(1000, 5)),
    columns=["A", "B", "C", "D", "E"],
)

# Divide os dados: 800 para treino e 200 para previsão
train_data = values[:800]
predict_data = values[800:]

print("Dados de Treino e Previsão Gerados.")
print(f"Total de amostras para treino: {len(train_data)}")
print(f"Total de amostras para previsão: {len(predict_data)}")

# 2. Criação e Treinamento do Modelo
# Define a estrutura da rede bayesiana
model = DiscreteBayesianNetwork(
    [("A", "B"), ("C", "B"), ("C", "D"), ("B", "E")]
)

# Treina o modelo: calcula as probabilidades (CPDs) a partir dos dados de treino
model.fit(train_data)
print("\nModelo treinado (CPDs ajustadas aos dados).")

# 3. Preparação dos Dados de Previsão (Evidência)
# Cria uma cópia dos dados e remove a coluna 'E'. 
# O objetivo é prever 'E' com base nos valores de A, B, C e D.
predict_data = predict_data.copy()
predict_data.drop("E", axis=1, inplace=True)
print("Variável 'E' removida dos dados de previsão (Evidência pronta).")

# 4. Configuração e Execução da Previsão
# Configura os parâmetros para a inferência aproximada (amostragem)
approx_inf_parameters = {"n_samples": int(1e3), "seed": 42}

# Realiza a previsão:
# - 'predict_data' são as evidências (A, B, C, D).
# - 'algo=ApproxInference' usa inferência por amostragem.
# - '**approx_inf_parameters' passa o número de amostras e a semente.
y_pred = model.predict(
    predict_data, algo=ApproxInference, **approx_inf_parameters
)

# 5. Imprime o Resultado da Previsão
print("\n--- Resultados da Previsão para a Variável 'E' ---")
# Imprime a coluna 'E' do DataFrame de previsão resultante
print(y_pred["E"].head())