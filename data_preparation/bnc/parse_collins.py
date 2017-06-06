import os
import operator
from collections import defaultdict

from ...lib import utils, corpus
from ..parse.parse import parse_file

SETTINGS = utils.read_settings()
TMP_FILE = 'tmp_pos.txt'

parser_path = SETTINGS.get('paths', 'parser')
parse_log = utils.set_logging('bnc-parse.log')

train_tag_path = SETTINGS.get('paths', 'dataBncTagTrain')
train_parse_path = SETTINGS.get('paths', 'dataBncParsedTrain')

chars = defaultdict(int)
counter = 0
for r, ds, fs in os.walk(train_tag_path):
    ds.sort()
    fs.sort()
    for f in fs:
        in_path = os.path.join(r, f)
        out_path = os.path.join(train_parse_path, f)

        parse_file(parser_path, in_path, out_path, parse_log)

#for ch in sorted(chars.items(), key=operator.itemgetter(1), reverse=True):
#    print ch[0], ch[1]

# remove tmp file at the end:
try:
    os.remove(TMP_FILE)
except OSError:
    pass


