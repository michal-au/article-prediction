from df_transform import LIST_FEATURES
import numbers
import pandas
import scipy
import numpy as np


# transform a list of elements (old_feature_name) into individual binary features
# in the form new_prefix_element
def resolve_list_features(df):
    print "transforming features to dicts ..."
    dict_data = []
    for row in df.to_dict(orient='records'):
        for feature, prefix in LIST_FEATURES:
            if row.get(feature):
                for val in row[feature]:
                    row[prefix + '_' + val] = True
            row.pop(feature, None)

        dict_data.append(row)
    print "... done"
    return dict_data


def remove_nans_from_dicts(dict_data):
    print "removing Nones from dicts ..."
    for row in dict_data:
        for k in row.keys():
            if row[k] is None or row[k] == []:
                del row[k]
    print "... done"


def prepare_for_vectorizer(df):
    print "preparing data for vectorizer ..."
    categorical_features = []  # strings
    categorical_list_features = []  # lists of strings
    for feature_name in df.columns:
        if feature_name == 'a_non_article_det':
            categorical_features.append(feature_name)
            continue
        if feature_name == 'b_object_form':
            categorical_features.append(feature_name)
            continue
        if feature_name == 'b_subject_predicate':
            categorical_features.append(feature_name)
            continue
        valid_index = df[feature_name].first_valid_index()
        try:
            valid_data = df[feature_name][valid_index]
        except KeyError:
            print feature_name
            raise NameError
        while valid_data == []:
            valid_index = df[feature_name][valid_index + 1:].first_valid_index()
            valid_data = df[feature_name][valid_index]

        if isinstance(valid_data, numbers.Number) or isinstance(valid_data, (bool, np.bool_)):
            # numeral values are kept intact
            continue
        if isinstance(valid_data, (list, np.ndarray)) and len(valid_data) and isinstance(valid_data[0], numbers.Number):
            # embeddings values have special treatment later
            continue
        if isinstance(valid_data, basestring):
            categorical_features.append(feature_name)
            continue
        if isinstance(valid_data, list) and len(valid_data) and isinstance(valid_data[0], basestring):
            categorical_list_features.append(feature_name)
            continue

        print valid_data, type(valid_data)
        raise NameError("FEATURE: {} is of an unknown type".format(feature_name))

    dicts_data = df[categorical_features].to_dict(orient='records')
    for i, row in enumerate(df[categorical_list_features].to_dict(orient='records')):
        for feature_name in categorical_list_features:
            if row.get(feature_name):
                for val in row[feature_name]:
                    row[feature_name + '_' + val] = True
                row.pop(feature_name, None)
        dicts_data[i].update(row)

    remove_nans_from_dicts(dicts_data)

    df = df.drop(categorical_list_features, 1)
    df = df.drop(categorical_features, 1)
    print "... done"
    return df, dicts_data


def convert_bools(df):
    print "converting bools ..."
    for feature_name in df.columns:
        valid_index = df[feature_name].first_valid_index()
        try:
            valid_data = df[feature_name][valid_index]
        except KeyError:
            continue
        if isinstance(valid_data, (bool, np.bool_)):
            print feature_name
            df[feature_name] = np.where(df[feature_name] == True, 1, 0)
            #df[feature_name] = np.where(df[feature_name] == True, 1, np.where(df[feature_name] is None, -1, 0))
            #df[feature_name] = np.where(df[feature_name] == True, 1, np.where(df[feature_name] == None, 0, 0))

    print "... done"
    return df