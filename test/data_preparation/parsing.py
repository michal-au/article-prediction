import os

from ...lib.utils import read_settings


SETTINGS = read_settings()


def test_parsing_log_exists():
    f_path = os.path.join(SETTINGS.get('paths', 'log'), 'parse.log')
    assert os.path.isfile(f_path), f_path


def test_parsed_files():
    orig_path = SETTINGS.get('paths', 'dataOrig')
    parsed_path = SETTINGS.get('paths', 'dataParsed')

    broken_files = []
    for r, d, f in sorted(os.walk(orig_path)):
        for f_name in sorted(f):
            orig_file_path = os.path.join(r, f_name)
            parsed_file_path = os.path.join(parsed_path, os.path.basename(os.path.normpath(r)), f_name)
            # parsed file exists:
            assert os.path.isfile(parsed_file_path)
            # parsed file is non-empty:
            if os.stat(parsed_file_path).st_size == 0:
                broken_files.append(parsed_file_path)
    assert not broken_files, broken_files
