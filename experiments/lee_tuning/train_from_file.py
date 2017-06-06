"""
Runs model training and evaluation based on an input file

file format:
each lines represents a single train-test run and should have the same format as experiment results, eg:
accuracy	model	call	features	train data	test data	time
0.8788007055	LogisticRegression	LogisticRegression(solver='lbfgs', multi_class='multinomial', C=10)	0 1 10 11 12 2 3 4 5 6 7 8 9	data/features/tun_train.pkl	data/features/tun_test.pkl	2016-07-05 15:51:22

only 'call' 'features' 'train data' and 'test data' columns are considered

Input: command-line argument, see the Makefile
Output: written into logs/experiments/model_results/penn/<model-name>.csv
"""

import sys
import csv

from ...lib.features import abbrevs_to_feature_names
#from ...lib.train_model import train_model
from ...lib.train_model_on_postprocessed_features import train_model


if __name__ == '__main__':
    if not sys.argv[1]:
        raise NameError("No File!")

    lines = []
    with open(sys.argv[1], 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            lines.append(row)

    l = len(lines) - 1
    for idx, line in enumerate(lines[1:], start=1):
        print("---------------------------------------------------------------------------")
        print("{0:0.2f}".format(idx / float(l)))
        print("---------------------------------------------------------------------------")

        _, _, _, model_call, features, train_dataset_name, test_dataset_name, _ = line
        features = abbrevs_to_feature_names([abb for abb in features.split()])
        train_model(
            features,
            model_call,
            train_dataset_name,
            test_dataset_name
        )
