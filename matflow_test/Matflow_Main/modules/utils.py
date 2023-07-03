
def get_variables(data, add_hypen=False):
    variables = data.columns.to_list()
    if add_hypen:
        variables.insert(0, "-")

    return variables


def get_categorical(data, add_hypen=False):
    cat_var = data.loc[:, data.dtypes == 'object'].columns.to_list()

    if add_hypen:
        cat_var.insert(0, "-")

    return cat_var


def get_numerical(data, add_hypen=False):
    num_var = data.loc[:, data.dtypes != 'object'].columns.to_list()

    if add_hypen:
        num_var.insert(0, "-")

    return num_var


def get_low_cardinality(data, max_unique=10, add_hypen=False):
    variables = data.loc[:, (data.nunique() <= max_unique)].columns.to_list()

    if add_hypen:
        variables.insert(0, "-")

    return variables


def get_null(data):
    null_var = data.loc[:, data.isna().sum() > 0].columns.to_list()

    return null_var


def get_dtypes(data):
    dtypes = data.dtypes.values.astype(str)

    return dtypes


def get_nunique(data, column=None):
    n_unique = data.nunique().to_list()
    if column:
        idx = data.columns.get_loc(column)
        n_unique = n_unique[idx]

    return n_unique



def split_xy(data, target_var):
    X = data.drop(target_var, axis=1)
    y = data[target_var]

    return X, y



def get_blank_column(data):
    columnsToDrop = []
    for column in data.columns:
        if data[column].isnull().values.all():
            columnsToDrop.append(column)
    return columnsToDrop
