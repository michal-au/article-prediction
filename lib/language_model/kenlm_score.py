import os
from subprocess import Popen, PIPE, STDOUT


KENLM = '/home/michal/kenlm/build/bin'


def zeros_score_candidates(model_path, a, b, c):
    """
    Scores three candidates (used in feature extraction)
    """
    text = '\n'.join((a, b, c)) + '\n'

    p = Popen([os.path.join(KENLM, 'query'), '-v', 'sentence', model_path], stdout=PIPE, stdin=PIPE, stderr=PIPE)
    stdin, stderr = p.communicate(input=text)

    return [float(chunk.split(' ')[0]) for chunk in stdin.split('Total: ')[1:]]


def score_candidates(model_path, a, b, c):
    """
    Scores three candidates (used in feature extraction)
    """
    perplex = []
    for candidate in (a, b, c):
        p = Popen([os.path.join(KENLM, 'query'), '-v', 'summary', model_path], stdout=PIPE, stdin=PIPE, stderr=PIPE)
        stdout, stderr = p.communicate(input=candidate + '\n')
        perplex.append(
            -float(stdout.split('Perplexity including OOVs:\t')[1].split('\n')[0])
        )
    return perplex
