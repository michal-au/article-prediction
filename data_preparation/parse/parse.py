import os
import subprocess

from ...lib import utils, corpus


SETTINGS = utils.read_settings()
TMP_FILE = 'tmp_pos.txt'


def parse_file(parser_path, old_file, new_file, log):
    """
    use the collins parser in the parser_path to parse the
    old_file producing the new file.

    :param parser_path: the path to the collins parser
    :type parser_path: str
    :param old_file: the path to file to be parsed
    :type old_file: str
    :param new_file: the path to file to be created
    :type new_file: str
    :param log: a configured logging module
    :type log: module 'logging'
    """

    utils.create_dir_for_file(new_file)

    prepare_pos_file(old_file, TMP_FILE)

    call = ' '.join([
        "gunzip -c",
        os.path.join(
            parser_path,
            'models/model3/events.gz'
        ),
        '|',
        os.path.join(
            parser_path,
            'code/parser'
        ),
        TMP_FILE,
        os.path.join(
            parser_path,
            'models/model3/grammar'
        ),
        "10000 1 1 1 1",
        "|",
        "code/data_preparation/lib/convertParses.prl",
        ">",
        new_file,
    ])
    info = 'PARSING: {} -> {}\n{}'.format(
        old_file,
        new_file,
        subprocess.check_output(call, shell=True, stderr=subprocess.STDOUT)
    )
    log.info(info)


def prepare_pos_file(f_path, new_path):
    """
    Transforms a pos file (tagged by mxpost-tagger into a file with the format expected by collins parser.

    :param f_path: the path to pos file to be transformed
    :type old_file: str
    :param new_file: the path to the transformed file to be created
    :type new_file: str
    """
    f = open(f_path, 'r')
    lines = []
    for line in f:
        line = line.rstrip()
        if not line:
            continue
        if line not in ['', ' ']:
            line = line.replace('_', ' ')
            line = line.replace(r'/', r'\/')
            words = line.split()
            words = [word for word in words if word != '']
            line = str(len(words)/2)+' '+' '.join(words)
        lines.append(line)
    utils.print_list_to_file(lines, new_path)


if __name__ == '__main__':
    parse_log = utils.set_logging('parse.log')
    old_path = SETTINGS.get('paths', 'dataPOS')
    new_path = SETTINGS.get('paths', 'dataParsed')
    parser_path = SETTINGS.get('paths', 'parser'),

    corpus.walk_and_transform(parse_file, parser_path, old_path, new_path, parse_log)

    # remove tmp file at the end:
    try:
        os.remove(TMP_FILE)
    except OSError:
        pass
