from pgmpy.utils import get_example_model
# É preciso importar a classe DiscreteBayesianNetwork para usar o método .load()
from pgmpy.models import DiscreteBayesianNetwork 

# 1. Carrega o modelo de exemplo "alarm"
alarm = get_example_model("alarm")
print("1. Modelo 'alarm' carregado (objeto em memória).")

# 2. Salva o modelo em um arquivo
# O modelo é serializado no formato BIF e salvo no arquivo "alarm.bif".
alarm.save("alarm.bif", filetype="bif")
print("2. Modelo salvo no arquivo 'alarm.bif' com sucesso.")

# 3. Carrega o modelo a partir do arquivo
# O método estático .load() lê o arquivo e cria uma nova instância da rede.
alarm_model = DiscreteBayesianNetwork.load("alarm.bif", filetype="bif")

# 4. Confirmação (mostrando que o modelo carregado é funcional)
print("3. Modelo carregado novamente a partir do arquivo.")
print(f"Nome do modelo carregado: {alarm_model.name}")
print(f"Número de nós no modelo carregado: {len(alarm_model.nodes())}")