import os

from ...lib.utils import read_settings


def process_orig_file(fPath):
    """
    count sentences in original PTB files based only on brackets
    """
    sentence_count = 0
    bracket_stack = 0
    with open(fPath, 'r') as f:
        for line in f:
            for char in line:
                if char == '(':
                    bracket_stack += 1
                elif char == ')':
                    bracket_stack -= 1
                    if bracket_stack == 0:
                        sentence_count += 1
        assert bracket_stack == 0

    return sentence_count


def process_raw_file(f_path):
    sentence_count = 0
    with open(f_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                sentence_count += 1
    return sentence_count


def process_pos_file(f_path):
    return process_raw_file(f_path)


def process_parsed_file(f_path):
    return process_raw_file(f_path)


def test_sentence_counts():
    broken_files = []

    settings = read_settings()
    orig_path = settings.get('paths', 'dataOrig')
    raw_path = settings.get('paths', 'dataRaw')
    pos_path = settings.get('paths', 'dataPOS')
    parsed_path = settings.get('paths', 'dataParsed')

    orig_sum = raw_sum = pos_sum = parsed_sum = 0
    for r, d, f in sorted(os.walk(orig_path)):
        for f_name in sorted(f):
            orig_file_path = os.path.join(r, f_name)
            raw_file_path = os.path.join(raw_path, os.path.basename(os.path.normpath(r)), f_name)
            pos_file_path = os.path.join(pos_path, os.path.basename(os.path.normpath(r)), f_name)
            parsed_file_path = os.path.join(parsed_path, os.path.basename(os.path.normpath(r)), f_name)

            orig = process_orig_file(orig_file_path)
            raw = process_raw_file(raw_file_path)
            pos = process_pos_file(pos_file_path)
            parsed = process_orig_file(parsed_file_path)
            if not orig == raw == pos == parsed:
                broken_files.append((f_name, orig, raw, pos, parsed))

            # get total number of sentences in each file type:
            orig_sum += orig
            raw_sum += raw
            pos_sum += pos
            parsed_sum += parsed_sum

    assert orig_sum == raw_sum == pos_sum

    # Collins parser does not parse some longish sentences. Even the provided
    # sec23 example is missing two sentences compared to the sec23.tagged file
    assert parsed_sum < orig_sum
    assert len([b_file[0] for b_file in broken_files if 'wsj_23' in b_file[0]]) == 2
