import os

from ...lib import utils


SETTINGS = utils.read_settings()


if __name__ == '__main__':
    f_path = os.path.join(SETTINGS.get('paths', 'log'), 'parse.log')
    assert os.path.isfile(f_path), f_path
    with open(f_path, 'r') as f:
        for line in f:
            if not (
                line.startswith('Hash table') or
                line.startswith('NUMSENTENCES') or
                line.endswith(' - INFO - Initialised lexicons\n') or
                not line or
                line in ('Initialised grammar\n', 'Loaded non-terminals\n', 'Loaded lexicon\n', 'Loaded grammar\n', '\n')
            ):
                print line.strip()
