import os

from ...lib.utils import read_settings, set_logging
from .parse import parse_file


SETTINGS = read_settings()


def get_files_with_long_sentences():
    """
    Gets the name of files that contain a sentence longer than 120 words
    (120 words is the default limit for collins parser)
    """
    raw_path = SETTINGS.get('paths', 'dataRaw')

    # storing long sentences: {length: (file, [sentence words])}
    long_sentences = {}

    for r, d, f in sorted(os.walk(raw_path)):
        for f_name in sorted(f):
            f_path = os.path.join(r, f_name)
            with open(f_path) as f:
                for line in f:
                    line = line.split()
                    if len(line) > 120:
                        long_sentences[len(line)] = (f_path, ' '.join(line))

    return [long_sentences[length][0] for length in sorted(long_sentences, reverse=True)]


def parse_error_files():
    """
    Parse the remaining two files that contained too long sentences and
    could not be parsed with the original collins parser
    """
    repair_log = set_logging('parse-repair.log')

    # path to the temporarily compiled parser (to parse sentences with too many words):
    parser_path = os.path.join(SETTINGS.get('paths', 'code'), 'data_preparation/parse/tmp_parser')

    # the dir to save the parsed files:
    new_path = SETTINGS.get('paths', 'dataParsed')
    old_path = SETTINGS.get('paths', 'dataPOS')

    for old_f in get_files_with_long_sentences():
        f_name = os.path.join(*os.path.normpath(old_f).split('/')[-2:])
        old_f = os.path.join(old_path, f_name)
        new_f = os.path.join(new_path, f_name)
        parse_file(parser_path, old_f, new_f, repair_log)


if __name__ == '__main__':
    parse_error_files()
