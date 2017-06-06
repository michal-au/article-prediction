import os
import pandas as pd

from code.lib import utils


TEST_PORTION = 0.15
SETTINGS = utils.read_settings()

DATA_SIZE = 400000


def dataset_path(teset, data_size=DATA_SIZE):
    return os.path.join(SETTINGS.get('paths', 'dataFeaturesBncLast'), '{}-{}.pkl'.format(teset, data_size))


if __name__ == '__main__':
    df = pd.read_pickle(os.path.join(SETTINGS.get('paths', 'dataFeaturesBncLast'), 'bnc-train-new-B.pkl'))
    df = df.sample(int(DATA_SIZE + DATA_SIZE*TEST_PORTION), random_state=0)
    test = df[:int(DATA_SIZE * TEST_PORTION)]
    fake_heldout = test[:2]
    train = df[int(DATA_SIZE * TEST_PORTION):]
    train.to_pickle(dataset_path('train'))
    test.to_pickle(dataset_path('test'))
    fake_heldout.to_pickle(dataset_path('heldout'))
