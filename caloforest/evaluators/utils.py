import copy
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def undummify_(df, prefix_sep="_"):
    cols2collapse = {
        item.split(prefix_sep)[0]: (prefix_sep in item) for item in df.columns
    }
    series_list = []
    for col, needs_to_collapse in cols2collapse.items():
        if needs_to_collapse:
            undummified = (
                df.filter(like=col + prefix_sep)
                .idxmax(axis=1)
                .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
                .rename(col)
            )
            series_list.append(undummified)
        else:
            series_list.append(df[col])
    undummified_df = pd.concat(series_list, axis=1)
    return undummified_df


def undummify(X, df_names_before, df_names_after):
    df = pd.DataFrame(X, columns = df_names_after) # to Pandas
    df = undummify_(df)[df_names_before]
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.to_numpy()
    return df


def dummify(X, cat_indexes, divide_by=0, drop_first=False):
    df = pd.DataFrame(X, columns = [str(i) for i in range(X.shape[1])]) # to Pandas
    df_names_before = df.columns
    for i in cat_indexes:
        df = pd.get_dummies(df, columns=[str(i)], prefix=str(i), dtype='float', drop_first=drop_first)
        if divide_by > 0: # needed for L1 distance to equal 1 when categories are different
            filter_col = [col for col in df if col.startswith(str(i) + '_')]
            df[filter_col] = df[filter_col] / divide_by
    df_names_after = df.columns
    df = df.to_numpy()
    return df, df_names_before, df_names_after


def minmax_scale_dummy(X_train, X_test, cat_indexes=[], mask=None, divide_by=2):
    X_train_ = copy.deepcopy(X_train)
    X_test_ = copy.deepcopy(X_test)
    # normalization of continuous variables
    scaler = MinMaxScaler()
    if len(cat_indexes) != X_train_.shape[1]: # if all variables are categorical, we do not scale-transform
        not_cat_indexes = [i for i in range(X_train_.shape[1]) if i not in cat_indexes]
        scaler.fit(X_train_[:, not_cat_indexes])

        #Transforms
        X_train_[:, not_cat_indexes] = scaler.transform(X_train_[:, not_cat_indexes])
        X_test_[:, not_cat_indexes] = scaler.transform(X_test_[:, not_cat_indexes])

    # One-hot the categorical variables (>=3 categories)
    df_names_before, df_names_after = None, None
    n = X_train.shape[0]
    if len(cat_indexes) > 0:
        X_train_test, df_names_before, df_names_after = dummify(np.concatenate((X_train_, X_test_), axis=0), cat_indexes, divide_by=divide_by)
        X_train_ = X_train_test[0:n,:]
        X_test_ = X_train_test[n:,:]

    # We get the new mask now that there are one-hot features
    if mask is not None:
        if len(cat_indexes) == 0:
            return X_train_, X_test_, mask, scaler, df_names_before, df_names_after
        else:
            mask_new = np.zeros(X_train_.shape)
            for i, var_name in enumerate(df_names_after):
                if '_' in var_name: # one-hot
                    var_ind = int(var_name.split('_')[0])
                else:
                    var_ind = int(var_name)
                mask_new[:, i] = mask[:, var_ind]
            return X_train_, X_test_, mask_new, scaler, df_names_before, df_names_after
    else:
        return X_train_, X_test_, scaler, df_names_before, df_names_after


def minmax_scale(X_train, X_test, cat_indexes=[]):
    X_train_ = copy.deepcopy(X_train)
    X_test_ = copy.deepcopy(X_test)
    # normalization of continuous variables
    scaler = MinMaxScaler()
    not_cat_indexes = [i for i in range(X_train_.shape[1]) if i not in cat_indexes]
    # Fit
    scaler.fit(X_train_[:, not_cat_indexes])

    #Transforms
    X_train_[:, not_cat_indexes] = scaler.transform(X_train_[:, not_cat_indexes])
    X_test_[:, not_cat_indexes] = scaler.transform(X_test_[:, not_cat_indexes])

    return X_train_, X_test_, scaler
