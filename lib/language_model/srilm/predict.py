import operator
import os
import subprocess
from ...utils import read_settings
from ...Constants import Constants


SETTINGS = read_settings()
MODEL_PATH = SETTINGS.get('paths', 'modelLM')


def pick_article_for_position(tokens, idx, model_name, order):
    # type: (List[str], int, str, str) -> str
    """
    :param tokens: a sentence as a list of tokens
    :param idx: index of the position we want to predict
    :param model_name: name of the language model
    :param order: order of the language model
    :return: most likely article for the given position in the sentence
    """

    # the, a, an:
    articles = (
        Constants.article_definite_token,
        Constants.article_indefinite_bucket_token,
        Constants.no_article_token
    )

    sents_to_test = []
    for article in articles:
        tokens[idx] = article
        sents_to_test.append(' '.join(tokens))

    call = ' '.join([
        'echo', '"{}"'.format('\n'.join(sents_to_test)), '|',
        '$HOME/srilm/bin/i686-m64/ngram',
        '-lm', os.path.join(MODEL_PATH, model_name),
        '-order', order,
        '-ppl -',
        '-debug 2'
    ])
    output = subprocess.check_output(call, shell=True, stderr=subprocess.STDOUT)

    """ SAMPLE OUTPUT:
    reading 7277 1-grams
    ....
    p( </s> | <unk> ...) 	= [1gram] 0.001207558 [ -2.918092 ]
    1 sentences, 8 words, 1 OOVs
    0 zeroprobs, logprob= -12.5671 ppl= 37.22948 ppl1= 62.4165

    a/an i <zero> want <zero> to <zero> vomit
        ...
        p( </s> | <unk> ...) 	= [1gram] 0.001207558 [ -2.918092 ]
    1 sentences, 8 words, 2 OOVs
    0 zeroprobs, logprob= -11.2002 ppl= 39.81334 ppl1= 73.56987

    <zero> i <zero> want <zero> to <zero> vomit
        ...
        p( </s> | <unk> ...) 	= [1gram] 0.001207558 [ -2.918092 ]
    1 sentences, 8 words, 1 OOVs
    0 zeroprobs, logprob= -9.308137 ppl= 14.57193 ppl1= 21.36652

    file -: 3 sentences, 24 words, 4 OOVs
    0 zeroprobs, logprob= -33.07543 ppl= 27.41967 ppl1= 45.05797
    """
    probs = [float(chunk.split(' ')[0]) for chunk in output.split('logprob= ')[1:-1]]
    max_index, max_value = max(enumerate(probs), key=operator.itemgetter(1))

    return articles[max_index]


def predict_sentence(tokens, model_name, model_order):
    print " ".join(tokens)
    for idx, token in enumerate(tokens):
        if token == Constants.no_article_token:
            tokens[idx] = pick_article_for_position(tokens, idx, model_name, model_order)
    print " ".join(tokens)


# def get_string_prob(string, model_name, order):
#     call = ' '.join([
#         'echo', '"{}"'.format(string), '|',
#         '$HOME/srilm/bin/i686-m64/ngram',
#         '-lm', os.path.join(MODEL_PATH, model_name),
#         '-order', order,
#         '-ppl -',
#         '-debug 2'
#     ])
#     output = subprocess.check_output(call, shell=True)
#     return float(output.split('logprob= ')[1].split(' ')[0])
