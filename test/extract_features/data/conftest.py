import pandas


def is_nonempty(df_f):
    return all(df_f.ravel())


def has_only_values(df_f, values):
    return set(pandas.unique(df_f.ravel())) == values


def is_boolean(df_f):
    return has_only_values(df_f, {True, False})

