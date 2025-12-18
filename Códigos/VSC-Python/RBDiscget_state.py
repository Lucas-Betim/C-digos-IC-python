from pgmpy.utils import get_example_model

# 1. Carrega o modelo de exemplo "asia"
model = get_example_model("asia")

# 2. Define o cenário (estado) que queremos calcular a probabilidade
# Os estados são: 'either' é 'no', 'tub' é 'no', 'xray' é 'yes', 'bronc' é 'no'.
estados = {"either": "no", "tub": "no", "xray": "yes", "bronc": "no"}

# 3. Calcula e imprime a probabilidade conjunta desse cenário
probabilidade = model.get_state_probability(estados)

print("--- Cálculo da Probabilidade Conjunta ---")
print(f"Cenário (Estados): {estados}")
print(f"A probabilidade desse cenário ocorrer é: {probabilidade}")