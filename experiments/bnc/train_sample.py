import os
import pandas

from ...lib.utils import read_settings
from .tune import features

SAMPLE_SIZE_FROM = 0
SAMPLE_SIZE_TO = 1000000
SAMPLE_SIZE = 2000000  # size of penn training data: 263008

# a: 3197470, b: 1477340, c: 3998973, d: 0, e: 1517917, f: 1816307, g: 1667519, h: 4172346, j: 1011471, k: 1426937
TOTAL_ROWS = 3197470 + 1477340 + 3998973 + 1517917 + 1816307 + 1667519 + 4172346 + 1011471 + 1426937

SETTINGS = read_settings()
feature_path = SETTINGS.get('paths', 'dataFeatures')

features.extend(['Y_article', '_coordinates'])


print "GETTING SAMPLES:"
for r, ds, fs in os.walk(feature_path):
    ds.sort()
    fs.sort()
    fs = [f for f in fs if f.startswith('bnc-train-')]
    for f in fs:
        f_path = os.path.join(r, f)
        print f_path
        df = pandas.read_pickle(f_path)
        df[features].sample(n=int(SAMPLE_SIZE_TO * (len(df)/float(TOTAL_ROWS))), random_state=243).to_pickle(
            os.path.join(feature_path, 'bnc-train_SAMPLE_{}.pkl'.format(f[-5]))
        )


print "JOINING SAMPLES:"
df = pandas.DataFrame()
for r, ds, fs in os.walk(feature_path):
    ds.sort()
    fs.sort()
    fs = [f for f in fs if f.startswith('bnc-train_SAMPLE')]
    for f in fs:
        f_path = os.path.join(r, f)
        print f_path
        df_part = pandas.read_pickle(f_path)
        df = df.append(df_part)

print len(df)
df.to_pickle(os.path.join(feature_path, 'bnc-train{}.pkl'.format(SAMPLE_SIZE)))
