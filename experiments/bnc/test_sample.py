import os
import pandas

from ...lib.utils import read_settings


SETTINGS = read_settings()
feature_path = SETTINGS.get('paths', 'dataFeatures')

df_full = pandas.read_pickle(os.path.join(feature_path, 'bnc-test.pkl'))
df = df_full.sample(n=len(df_full)/10, random_state=243)
print len(df_full), '>>', len(df)

df.to_pickle(os.path.join(feature_path, 'bnc-test{}.pkl'.format(len(df))))


