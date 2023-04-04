from typing import List

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE




def run_dr(
        df,
        remove:List =['Country name', 'Standard error of ladder score', 'Ladder score']
):
    # pca = PCA(n_components=2)
    pca = TSNE(n_components=2)
    keep_cols = [c for c in df.columns if c not in remove]
    df5 = df[keep_cols]
    df6 = pd.get_dummies(df5, prefix_sep='::')

    projected = pca.fit_transform(df6)
    # pca.get_precision()
    # pca.get_covariance()

    return pca, projected
