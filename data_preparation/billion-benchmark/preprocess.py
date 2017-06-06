"""
THIS FILE IS REPLACED BY preprocess_sentence.py

tohle nebylo pouzity a da se to smazat (doufejme)
"""
import sys
import os

from ...lib.Constants import Constants
from ...lib.utils import read_settings
from ...lib.language_model.utils.sentence_preprocess import process_nbs, replace_indef_tokens_by_bucket
SETTINGS = read_settings()


def preprocess_sentence(f, fw):

    for l in f:
        l = l.strip()
        if not l:
            continue

        sent = []

        for token in l.split(' '):
            token = token.lower()
            token = process_nbs(token)
            token = replace_indef_tokens_by_bucket(token)

            ## TODO: smazat -- puvodni napad byl cpat <zero> na kazdy misto, kde se potencialne muze vyskytnout clen, diky perplexite to ale nepotrebujeme
            # if (
            #     (token in Constants.article_tokens) or
            #     (sent and sent[-1] in (Constants.article_indefinite_bucket_token, Constants.article_definite_token))
            # ):
            #     if token in Constants.article_indefinite_tokens:
            #         # 'a', 'an' -> 'a/an'
            #         token = Constants.article_indefinite_bucket_token
            #     sent.append(token)
            # else:
            #     sent.extend([Constants.no_article_token, token])
            sent.append(token)

        fw.write(' '.join(sent) + '\n')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        raise NameError('Give me a file to transform')

    fw = open(os.path.join(SETTINGS.get('paths', 'dataBenchmarkPreprocessed'), 'train-lc-nbs'), 'w')

    for input_f in sys.argv[1:]:
        print input_f
        with open(input_f, 'r') as f:
            preprocess_sentence(f, fw)
