import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Nomes das colunas para facilitar a leitura dos dados
column_names = ['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
                'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
                'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se',
                'concave_points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
                'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst',
                'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst']

# ---------- Funções do Perceptron ----------
def predict(row, weights):
    activation = weights[0]  # bias
    for i in range(len(row)):
        activation += weights[i + 1] * row[i]
    return 1.0 if activation >= 0.0 else 0.0

def train_weights(train_x, train_y, learning_rate, epochs):
    weights = [0.0 for _ in range(len(train_x[0]) + 1)]  # +1 para o bias
    for epoch in range(epochs):
        total_error = 0
        for i in range(len(train_x)):
            prediction = predict(train_x[i], weights)
            error = train_y[i] - prediction
            total_error += abs(error)
            weights[0] += learning_rate * error  # bias
            for j in range(len(train_x[i])):
                weights[j + 1] += learning_rate * error * train_x[i][j]
        print(f"Época {epoch+1}, Erro Total: {total_error:.2f}")
    return weights

# ---------- Carregando o Dataset ----------
def load_data():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'
    df = pd.read_csv(url, header=None, names=column_names)
    
    # Remover colunas desnecessárias
    df = df.drop(['id'], axis=1)
    
    # Converter 'M' e 'B' para 1 e 0
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    
    # Separar atributos e rótulo
    X = df.drop('diagnosis', axis=1).values
    y = df['diagnosis'].values

    # Normalizar
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    return X, y

# ---------- Execução ----------
if __name__ == "__main__":
    print("Carregando dados...")
    X, y = load_data()

    # Divisão dos dados em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print("Treinando o Perceptron...")
    learning_rate = 0.01
    epochs = 50
    weights = train_weights(X_train, y_train, learning_rate, epochs)

    print("\nPesos finais:", weights)

    print("\nTestando acurácia no conjunto de teste:")
    correct = 0
    for i in range(len(X_test)):
        prediction = predict(X_test[i], weights)
        if prediction == y_test[i]:
            correct += 1
    accuracy = correct / len(X_test)
    print(f"Acurácia: {accuracy*100:.2f}%")