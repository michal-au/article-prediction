import argparse

from ...lib.utils import read_settings
from ...lib.language_model.utils.sentence_preprocess import process_nbs, replace_indef_tokens_by_bucket
SETTINGS = read_settings()


def preprocess_sentence(f, args):

    for l in f:
        l = l.strip()
        if not l:
            continue

        sent = []

        for token in l.split(' '):
            #token = token.lower()
            if args.nbs:
                token = process_nbs(token)
            if args.cls3:
                token = replace_indef_tokens_by_bucket(token)
            sent.append(token)

        print ' '.join(sent)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Preprocess text before it is used for language model')
    parser.add_argument('files', type=str, nargs='+', help='a source file')
    parser.add_argument('--nbs', action='store_true', help='replace numbers with the "<number>" token')
    parser.add_argument('--cls3', action='store_true', help='')
    args = parser.parse_args()

    print args

    for input_f in args.files:
        with open(input_f, 'r') as f:
            preprocess_sentence(f, args)
