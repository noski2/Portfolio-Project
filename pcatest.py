import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

df = pd.DataFrame(np.random.randint(0,100,size=(1200, 4000)))

pca = PCA(n_components= 81)
pca.fit(df.transpose())

print(pd.DataFrame(pca.components_).transpose())