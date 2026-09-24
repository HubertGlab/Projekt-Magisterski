import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

def cluster_variables_by_correlation(df, columns, method_corr="spearman", threshold_corr=0.9):
    
    data = df[columns].copy()
    
    corr_matrix = data.corr(method=method_corr).abs()
    
    dist_matrix = 1 - corr_matrix
    
    np.fill_diagonal(dist_matrix.values, 0)

    condensed_dist = squareform(dist_matrix.values, checks=False)
    
    linkage_matrix = linkage(condensed_dist, method="average")
    
    distance_cut = 1 - threshold_corr
    
    cluster_labels = fcluster(linkage_matrix, t=distance_cut, criterion="distance")
    
    cluster_map = pd.DataFrame({
        "variable": columns,
        "cluster": cluster_labels
    }).sort_values(["cluster", "variable"]).reset_index(drop=True)
    
    return corr_matrix, dist_matrix, linkage_matrix, cluster_map