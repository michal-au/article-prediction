import numpy as np

EMBEDDING_LENGTH = 300


def process_embeddings(df, features):
    print('processing embeddings ...')
    features = [f for f in features]
    df.index = xrange(len(df))

    added_features = []
    features_to_remove = []
    for feature_name in features:
        if feature_name.startswith('d_'):
            print(' FEATURE: {}'.format(feature_name))
            embeddings_length = len(df[feature_name][df[feature_name].last_valid_index()])
            dummy_list = np.array([None] * embeddings_length)

            print(' - joining vectors')
            vectors = []
            for i in xrange(len(df[feature_name])):
                if df[feature_name][i] is None:
                    vectors.append(dummy_list)
                else:
                    vectors.append(df[feature_name][i])
            vectors = np.vstack(vectors)

            print(' - transforming vectors')
            df = df.drop(feature_name, 1)
            for i in xrange(embeddings_length):
                added_features.append(feature_name + "_" + str(i))
                df[feature_name + "_" + str(i)] = vectors[:,  i]
                df[feature_name + "_" + str(i)] = df[feature_name + "_" + str(i)].astype(float)
                df[feature_name + "_" + str(i)] = df[feature_name + "_" + str(i)].fillna(df[feature_name + "_" + str(i)].mean())

            features_to_remove.append(feature_name)

    for feature_name in features_to_remove:
        features.remove(feature_name)
    features.extend(added_features)
    print('... done')
    return df, features


def postprocess_embeddings(df):
    """
    Nahrada predchozi fce process_embeddings(). Pokud to vyjde, predchozi nebude treba. Volame pro ulozeni embeddingu
    na disk jako crf matrix
    """
    df.index = xrange(len(df))
    embeddings_arrays = {}
    for feature_name in df.columns:
        if feature_name.startswith('d_'):
            dummy_list = np.array([None] * EMBEDDING_LENGTH)
            vectors = [
                dummy_list if df[feature_name][i] is None else df[feature_name][i]
                for i in xrange(len(df[feature_name]))
            ]
            array = np.array(vectors, dtype=np.float)

            # doplneni chybejicich hodnot prumerem ze sloupce:
            idxs = np.where(np.isnan(array))
            array[idxs] = np.take(np.nanmean(array, axis=0), idxs[1])
            embeddings_arrays[feature_name] = array

            df = df.drop(feature_name, 1)

    return df, embeddings_arrays
