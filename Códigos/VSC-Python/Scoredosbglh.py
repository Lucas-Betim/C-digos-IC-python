import pandas as pd
import numpy as np
from pgmpy.estimators import BDeu, K2, BIC
from pgmpy.models import DiscreteBayesianNetwork

# create random data sample with 3 variables, where Z is dependent on X, Y:
data = pd.DataFrame(np.random.randint(
    0, 4, size=(5000, 2)), columns=list('XY'))
data['Z'] = data['X'] + data['Y']

bdeu = BDeu(data, equivalent_sample_size=5)
k2 = K2(data)
bic = BIC(data)

model1 = DiscreteBayesianNetwork([('X', 'Z'), ('Y', 'Z')])  # X -> Z <- Y
model2 = DiscreteBayesianNetwork([('X', 'Z'), ('X', 'Y')])  # Y <- X -> Z

print(bdeu.score(model1))
print(k2.score(model1))
print(bic.score(model1))

print(bdeu.score(model2))
print(k2.score(model2))
print(bic.score(model2))

-13938.353002020234
-14329.194269073454
-14294.390420213556
-20906.432489257266
-20933.26023936978
-20950.47339067585
