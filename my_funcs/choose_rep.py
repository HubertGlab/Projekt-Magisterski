import pandas as pd


def choose_cluster_representatives(cluster_map, corr_matrix):

    two_vars = []
    representatives = []
    summary_rows = []
    
    for cluster_id, group_df in cluster_map.groupby("cluster"):
        variables = group_df["variable"].tolist()
        rep = None
        mean_distance = None
        mean_abs_corr = None
        if len(variables) == 1:
            rep = variables[0]
            mean_distance = 0.0
            mean_abs_corr = 1.0
        elif len(variables) == 2:
            two_vars.append((variables[0], variables[1]))
        else:
            sub_corr = corr_matrix.loc[variables, variables].copy()
            
            sub_dist = 1 - sub_corr
            
            mean_distance_series = (sub_dist.sum(axis=1)) / (len(variables) - 1)
            
            rep = mean_distance_series.idxmin()
            
            mean_distance = mean_distance_series.loc[rep]
            mean_abs_corr_series = (sub_corr.sum(axis=1) - 1) / (len(variables) - 1)
            mean_abs_corr = mean_abs_corr_series.loc[rep]
        
        if rep is not None:
            to_drop = [v for v in variables if v != rep]
            representatives.append(rep)
        
        summary_rows.append({
            "cluster": cluster_id,
            "n_variables": len(variables),
            "representative": rep,
            "variables": variables,
            "to_drop": to_drop if rep is not None else None,
            "representative_mean_distance": mean_distance,
            "representative_mean_abs_corr": mean_abs_corr
        })
    
    summary_df = (
        pd.DataFrame(summary_rows)
        .sort_values("cluster")
        .reset_index(drop=True)
    )
    
    return representatives, summary_df, two_vars