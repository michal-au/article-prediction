import numpy as np
#from ..additional_methods.predicate import object_verb_form
import cPickle as pickle
import os
from ...lib.utils import read_settings


def head_form_embeddings(model, head_form):
    try:
        return model[head_form]
    except KeyError:
        return None


def embed_poly(embed):
    # type: List[float]
    result = []
    if embed == [] or embed is None:
        return result
    for i in xrange(len(embed)):
        for j in xrange(i, len(embed)):
            result.append(embed[i] * embed[j])
    return tuple(result)



def embed_feature_selection(model, head_form, indicies):
    """

    :param model: word2vec embedding lookup
    :param head_form:
    :param indicies: list of indicies selected in feature selection step
    :return:
    """
    try:
        orig_emb = model[head_form]
        emb_length = len(orig_emb)  # 300
        emb = list(orig_emb) + [orig_emb[i]*orig_emb[j] for i in xrange(emb_length) for j in xrange(i, emb_length)]
        return [emb[idx] for idx in indicies]
    except KeyError:
        return None


# def object_form_embeddings(model, bnp):
#     verb_form = object_verb_form(bnp)
#     if not verb_form:
#         return None
#     verb_form = verb_form.split(' ')[-1]
#     return head_form_embeddings(model, verb_form)
#
#
# def avg_embeddings(model, words):
#     vectors = []
#     for word in words:
#         vector = head_form_embeddings(model, word)
#         if vector is not None:
#             vectors.append(vector)
#     if vectors:
#         return np.mean(vectors, axis=0)
#     else:
#         return None