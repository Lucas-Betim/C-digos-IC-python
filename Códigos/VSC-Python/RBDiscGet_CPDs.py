from pgmpy.utils import get_example_model
from pgmpy.factors.discrete import TabularCPD # Importação útil para fins de tipagem na saída

# 1. Carrega o modelo de exemplo "asia"
model = get_example_model("asia")

print("Modelo 'asia' carregado com sucesso.")

# 2. Obtém TODAS as CPDs do modelo
cpds = model.get_cpds()

print("\nTodas as Tabelas de Probabilidade Condicional (CPDs) do modelo:")
# Imprime a lista completa de objetos CPDs
print(cpds)
# Nota: O endereço de memória (ex: '0x7dbbd9bdbb80') mudará a cada execução.

# 3. Obtém a CPD de uma variável específica ("bronc")
cpd = model.get_cpds("bronc")

print("\nCPD da variável 'bronc':")
# Imprime o objeto CPD específico
print(cpd)

# 4. Opcional: Imprime a tabela de probabilidades para melhor visualização
print("\nDetalhes da Tabela de Probabilidade Condicional de 'bronc':")
print(cpd.values)