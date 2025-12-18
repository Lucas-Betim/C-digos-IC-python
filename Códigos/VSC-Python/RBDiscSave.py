from pgmpy.utils import get_example_model
from pgmpy.models import DiscreteBayesianNetwork 
import os

# Define o nome do arquivo para salvar o modelo
NOME_DO_ARQUIVO = "meu_modelo_asia.bif"

# 1. Carrega um modelo de exemplo ("asia")
modelo_original = get_example_model("asia")
print(f"1. Modelo 'asia' carregado em memória. Nós: {modelo_original.nodes()}")

# --- SALVANDO (save) ---
# 2. Salva o modelo no arquivo com o novo nome
modelo_original.save(NOME_DO_ARQUIVO, filetype="bif")
print(f"\n2. SUCESSO: Modelo salvo no arquivo '{NOME_DO_ARQUIVO}'.")

# --- CARREGANDO (load) ---
# 3. Carrega o modelo de volta para uma nova variável
modelo_carregado = DiscreteBayesianNetwork.load(NOME_DO_ARQUIVO, filetype="bif")

# 4. Confirmação
print(f"\n3. Modelo carregado novamente a partir do arquivo.")
print(f"Nós do modelo carregado: {modelo_carregado.nodes()}")
print(f"CPDs no modelo carregado: {len(modelo_carregado.get_cpds())}")

# 5. Limpeza (Removendo o arquivo criado para manter a pasta limpa)
os.remove(NOME_DO_ARQUIVO)
print(f"\nArquivo temporário '{NOME_DO_ARQUIVO}' removido.")