import pandas as pd
import numpy as np
from pgmpy.estimators.CITests import chi_square
from pgmpy.estimators import PC 

# 1. Criação de Dados com Dependências Específicas
data = pd.DataFrame(np.random.randint(0, 3, size=(2500, 8)), columns=list('ABCDEFGH'))
data['A'] = data['B'] + data['C']
data['H'] = data['G'] - data['A']
data['E'] = data['E'] * data['F']
for col in data.columns:
    data[col] = data[col].astype('category')

print("--- Resultados dos Testes Chi-Square (Versão Simplificada) ---")

# 2. Testes de Independência (Corrigido para capturar apenas o Booleano)

# Teste 1: B e H (Dependentes esperados)
is_independent1 = chi_square(X='B', Y='H', Z=[], data=data, significance_level=0.05)
print(f"B ⊥ H | []: Independente? {is_independent1} (Esperado: False)")

# Teste 2: B e E (Independentes esperados)
is_independent2 = chi_square(X='B', Y='E', Z=[], data=data, significance_level=0.05)
print(f"B ⊥ E | []: Independente? {is_independent2} (Esperado: True)")

# Teste 3: B e H condicional a A (Independentes esperados)
is_independent3 = chi_square(X='B', Y='H', Z=['A'], data=data, significance_level=0.05)
print(f"B ⊥ H | [A]: Independente? {is_independent3} (Esperado: True)")

# Teste 4: A e G (Independentes esperados)
is_independent4 = chi_square(X='A', Y='G', Z=[], data=data, significance_level=0.05)
print(f"A ⊥ G | []: Independente? {is_independent4} (Esperado: True)")

# Teste 5: A e G condicional a H (Dependentes esperados)
is_independent5 = chi_square(X='A', Y='G', Z=['H'], data=data, significance_level=0.05)
print(f"A ⊥ G | [H]: Independente? {is_independent5} (Esperado: False)")