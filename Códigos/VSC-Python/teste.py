# Função para fazer uma previsão com os pesos atuais
def predict(row, weights):
    activation = weights[0]  # bias
    for i in range(len(row)-1):  # -1 porque o último valor é o rótulo (0 ou 1)
        activation += weights[i + 1] * row[i]
    return 1.0 if activation >= 0.0 else 0.0  # função degrau

# Função para treinar os pesos
def train_weights(train, l_rate, n_epoch):
    weights = [0.0 for _ in range(len(train[0]))]  # inicializa pesos em 0
    for epoch in range(n_epoch):
        print(f'Época {epoch + 1}:')
        for row in train:
            prediction = predict(row, weights)
            error = row[-1] - prediction
            weights[0] += l_rate * error  # atualiza bias
            for i in range(len(row)-1):
                weights[i + 1] += l_rate * error * row[i]  # atualiza pesos
            print(f'  Entrada: {row[:-1]}, Esperado: {row[-1]}, Previsto: {prediction}, Erro: {error}')
    return weights

# Exemplo de conjunto de dados simples [entrada1, entrada2, saída]
# Queremos aprender a identificar se o ponto está acima ou abaixo da linha x + y = 1.5
dataset = [
    [0.5, 1.0, 0],
    [1.5, 0.5, 1],
    [1.0, 1.0, 1],
    [0.0, 0.5, 0],
    [0.8, 0.6, 1],
    [0.3, 0.2, 0]
]

# Configurações
taxa_aprendizado = 0.1
epocas = 10

# Treinar
pesos = train_weights(dataset, taxa_aprendizado, epocas)

# Mostrar os pesos finais
print("\nPesos finais aprendidos:")
print(pesos)

# Testar com nova entrada
entrada_nova = [1.0, 0.5]  # sem o rótulo
saida = predict(entrada_nova + [None], pesos)
print(f"\nPrevisão para entrada {entrada_nova}: Classe {int(saida)}")
