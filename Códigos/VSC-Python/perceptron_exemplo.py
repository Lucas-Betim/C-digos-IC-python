# -------------------------------------------------------------
# 1. Função de Previsão: calcula a saída do Perceptron
# -------------------------------------------------------------
def predict(row, weights):
    """Calcula a ativação e retorna a previsão (1.0 ou 0.0)."""
    activation = weights[0]  # O primeiro peso é o bias
    for i in range(len(row) - 1):
        activation += weights[i + 1] * row[i]
    return 1.0 if activation >= 0.0 else 0.0

# -------------------------------------------------------------
# 2. Função de Treinamento: ajusta os pesos do Perceptron
# -------------------------------------------------------------
def train_weights(train, l_rate, n_epoch):
    """
    Treina o Perceptron ajustando os pesos com base nos erros.
    l_rate: taxa de aprendizado
    n_epoch: número de épocas (passagens pelo conjunto de dados)
    """
    # Inicializa todos os pesos com zero
    weights = [0.0 for i in range(len(train[0]))]
    
    for epoch in range(n_epoch):
        for row in train:
            # Faz uma previsão com os pesos atuais
            prediction = predict(row, weights)
            # Pega a resposta correta
            expected = row[-1]
            # Calcula o erro
            error = expected - prediction
            
            # Ajusta o peso do bias (o primeiro peso)
            weights[0] = weights[0] + l_rate * error
            
            # Ajusta os pesos das entradas
            for i in range(len(row) - 1):
                weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
    return weights

# -------------------------------------------------------------
# 3. Algoritmo Perceptron Completo: junta tudo e faz as previsões
# -------------------------------------------------------------
def perceptron(train, test, l_rate, n_epoch):
    """Treina e testa o Perceptron em conjuntos de dados."""
    predictions = list()
    # Chama a função de treinamento para obter os pesos finais
    weights = train_weights(train, l_rate, n_epoch)
    
    # Usa os pesos aprendidos para fazer previsões no conjunto de teste
    for row in test:
        prediction = predict(row, weights)
        predictions.append(prediction)
    return predictions

# =============================================================
# Exemplo de Uso e Teste
# =============================================================
if __name__ == '__main__':
    # Conjunto de dados de exemplo (os 3 primeiros valores são as entradas, o último é a saída esperada)
    dataset = [[2.7810836, 2.550537003, 0],
               [1.465489372, 2.362125076, 0],
               [3.396561688, 4.400293529, 0],
               [1.38807019, 1.850220317, 0],
               [3.06407232, 3.005305206, 0],
               [7.627531214, 2.759262235, 1],
               [5.332441248, 2.088626775, 1],
               [6.922596716, 1.77106367, 1],
               [8.675418651, -0.24206863, 1],
               [7.673756466, 3.508563011, 1]]

    # Define os parâmetros do algoritmo
    l_rate = 0.1  # Taxa de aprendizado
    n_epoch = 5   # Número de épocas

    # Treina o Perceptron e faz as previsões (aqui, o conjunto de teste é o mesmo do treinamento para simplificar)
    predictions = perceptron(dataset, dataset, l_rate, n_epoch)

    # Exibe os resultados
    print("Previsões do Perceptron:")
    for i in range(len(dataset)):
        print('Esperado=%d, Previsto=%.1f' % (dataset[i][-1], predictions[i]))