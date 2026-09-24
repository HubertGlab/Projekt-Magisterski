import pandas as pd
import numpy as np

def hellwig_compensational(df, stim_or_destim, pattern=None, D0=None):

    deviations = pd.DataFrame(index=df.index)
    stim_cols = stim_or_destim["variable"][stim_or_destim["type"] == "stim"].to_list()
    destim_cols = stim_or_destim["variable"][stim_or_destim["type"] == "destim"].to_list()


    if pattern is None:
        pattern = {}
        for column in df.columns:
            if column in stim_cols:
                pattern[column] = df[column].max()
            elif column in destim_cols:
                pattern[column] = df[column].min()
        pattern = pd.Series(pattern)

    for column in pattern.index:
        if column in stim_cols:
            deviations[column] = pattern[column] - df[column]

        elif column in destim_cols:
            deviations[column] = df[column] - pattern[column]
            
    distance = deviations.sum(axis=1)

    if D0 is None:
        D0 = distance.mean() + 2 * distance.std()
    q = round(1 - (distance / D0), 4)

    return q, distance, deviations, pattern, D0


def hellwig_classic(df, stim_or_destim, pattern=None, D0=None):
    deviations = pd.DataFrame(index=df.index)

    stim_cols = stim_or_destim["variable"][stim_or_destim["type"] == "stim"].to_list()
    destim_cols = stim_or_destim["variable"][stim_or_destim["type"] == "destim"].to_list()

    if pattern is None:
        pattern = {}

        for column in df.columns:
            if column in stim_cols:
                pattern[column] = df[column].max()
            elif column in destim_cols:
                pattern[column] = df[column].min()

        pattern = pd.Series(pattern)

    for column in pattern.index:
        deviations[column] = df[column] - pattern[column]

    distance = np.sqrt((deviations ** 2).sum(axis=1))

    if D0 is None:
        D0 = distance.mean() + 2 * distance.std()

    q = (1 - (distance / D0)).round(4)

    return q, distance, deviations, pattern, D0


def hellwig_manhattan(df, stim_or_destim, pattern=None, D0=None):
    deviations = pd.DataFrame(index=df.index)

    stim_cols = stim_or_destim["variable"][stim_or_destim["type"] == "stim"].to_list()
    destim_cols = stim_or_destim["variable"][stim_or_destim["type"] == "destim"].to_list()

    if pattern is None:
        pattern = {}

        for column in df.columns:
            if column in stim_cols:
                pattern[column] = df[column].max()
            elif column in destim_cols:
                pattern[column] = df[column].min()

        pattern = pd.Series(pattern)

    for column in pattern.index:
        deviations[column] = df[column] - pattern[column]

    distance = deviations.abs().sum(axis=1)

    if D0 is None:
        D0 = distance.mean() + 2 * distance.std()

    q = (1 - (distance / D0)).round(4)

    return q, distance, deviations, pattern, D0