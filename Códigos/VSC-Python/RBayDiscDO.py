from pgmpy.utils import get_example_model

# 1. Carrega o modelo de exemplo "asia"
# Este modelo representa um problema de diagnóstico médico.
asia = get_example_model("asia")

print("--- Modelo Original ---")

# 2. Imprime as arestas (dependências) do modelo original
print("Arestas ANTES da intervenção (asia.edges()):")
print(asia.edges())

# 3. Simula a intervenção causal (do-calculus)
# A intervenção do(["bronc"]) é simular que a variável 'bronc'
# (bronquite) foi FORÇADA a um valor, quebrando suas conexões causais de entrada.
do_bronc = asia.do(["bronc"])

print("\n--- Modelo Após Intervenção do(['bronc']) ---")

# 4. Imprime as arestas do modelo após a intervenção
print("Arestas DEPOIS da intervenção (do_bronc.edges()):")
print(do_bronc.edges())