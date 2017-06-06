# should find out which dimensions of the embedding vectors are the most interesting for the classifier

# cesta k datum, z kterych ziskame word embeddingy
DF_PATH = '/home/michal/diplomka/code/data/features/penn/train.pkl'

import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif #, mutual_info_classif
from collections import defaultdict


CHUNK_SIZE = 4000


#1: nacti dataframe:
df = pd.read_pickle(DF_PATH)

scores = defaultdict(int)
#2: random sample, s vypustenim prazdnych hodnot:
#s = df[['d_head_form_embeddings', 'Y_article']].dropna().sample(n=1000, random_state=42)
df_shuffled = df[['d_head_form_embeddings', 'Y_article']].sample(n=df.shape[0], random_state=42).dropna()

for cycle_idx in xrange(10):
    print cycle_idx
    s = df_shuffled[cycle_idx*CHUNK_SIZE:(cycle_idx+1)*CHUNK_SIZE]
    #3: nove featury jako polynom. kombinaci word embeddingu:
    polys = pd.DataFrame.from_records(s.d_head_form_embeddings.apply(lambda x: [x[i]*x[j] for i in xrange(300) for j in xrange(i,300)]).values.tolist(), index=s.index)
    polys.columns = ['poly_' + str(i) for i in xrange(len(polys.columns))]
    #4: puvodni word embeddingy:
    orig_embeds = pd.DataFrame.from_records(s.d_head_form_embeddings.values.tolist(), index=s.index)
    orig_embeds.columns = ['orig_' + str(i) for i in xrange(len(orig_embeds.columns))]
    #5: spojeni:
    X = pd.concat([orig_embeds, polys], axis=1)
    #6: vybrani 300 nejlepsich featur
    y = s.Y_article
    f_selector_f = SelectKBest(f_classif, k=300)
    f_selector_f.fit(X, y)
    #f_selector_mi = SelectKBest(mutual_info_classif, k=300)
    #f_selector_mi.fit(X, y)
    # Indexy nejlepcich 300 featur:
    a = f_selector_f.scores_
    idxs_f = sorted(range(len(a)), key=lambda i: a[i])[-300:]
    #a = f_selector_mi.scores_
    #idxs_mi = sorted(range(len(a)), key=lambda i: a[i])[-300:]
    for idx_f in idxs_f:
        scores[idx_f] += 1
