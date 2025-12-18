from pgmpy.utils import get_example_model
from pgmpy.sampling import BayesianModelSampling
from pgmpy.models import BayesianNetwork # Importação necessária para o modelo alarm

# 1. Carrega o modelo de exemplo "alarm"
# Nota: O modelo 'alarm' é um exemplo complexo com 37 nós.
model = get_example_model("alarm")

# 2. Gera novos dados
# Cria um objeto de amostragem (sampler) para o modelo.
sampler = BayesianModelSampling(model)

# Gera 1000 novas amostras (dados simulados)
num_samples = 1000
data = sampler.forward_sample(num_samples)

print(f"Geradas {num_samples} novas amostras de dados.")

# 3. Atualiza os parâmetros do modelo
# Usa o novo 'data' para refinar as Tabelas de Probabilidade Condicional (CPDs)
# do 'model' (que é a funcionalidade do fit_update).
model.fit_update(data)

print("Parâmetros do modelo foram atualizados com sucesso usando os novos dados.")