import pandas
import os
import sys

from ...lib.train_model import train_model, train_svm_model
from ...lib.utils import read_settings

from .feature_sets import (
    basic,
    basic_modified,
    basic_modified_v2,
    basic_modified_v3,
    basic_modified_v4,
    basic_modified_v5,
    b_features,
    b_c_features,
    b_c_features_v2,
    b_c_features_v3,
    b_c_d_features,
    b_c_d_features_v2,
    features,
)


if __name__ == '__main__':
    only_console = True if len(sys.argv) > 1 and sys.argv[1] == 'console_only' else False
    cross_validation = True if len(sys.argv) > 2 and sys.argv[2] == 'cross_valid' else False

    settings = read_settings()
    path = settings.get('paths', 'dataFeatures')
    #df_train_path = os.path.join(path, "tun_train.pkl")
    #df_heldout_path = os.path.join(path, "tun_heldout.pkl")
    #df_test_path = os.path.join(path, "tun_test.pkl")
    #df_test_path_full = os.path.join(path, "tun_test_all_nps.pkl")

    df_train_path_simple = os.path.join(path, "penn/train.pkl")
    df_test_path_simple = os.path.join(path, "penn/test.pkl")
    df_heldout_path_simple = os.path.join(path, "penn/test.pkl")

    for ftrs in (b_c_d_features, ):
        for regular in ('l1', 'l2'):
            #for i in (1,):
            for i in (0.01, 0.1, 1, 1.5, 2):
                #call = "LogisticRegression(solver='lbfgs', multi_class='multinomial', C=1)"
                #call = "LogisticRegression(penalty='l2', solver='liblinear')"

                call = "LogisticRegression(penalty='{}', solver='liblinear', C={})".format(regular, i)
                #call = "LogisticRegression(solver='lbfgs', multi_class='multinomial', C={})".format(i)

                #call = "SGDClassifier(loss='modified_huber', penalty='elasticnet')"
                #call = "SGDClassifier(loss='log', penalty='l1', n_iter=5)"

                #call = "LogisticRegression(penalty='{}', solver='liblinear', C=1)".format(regular)
                train_model(
                    ftrs,
                    call,
                    df_train_path_simple,
                    df_test_path_simple,
                    only_console=only_console,
                    cross_validation=cross_validation
                )

            #train_svm_model(features, df_train_path, df_heldout_path, only_console=True)