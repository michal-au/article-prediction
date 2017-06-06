import codecs
import os
from nltk.parse.stanford import StanfordParser

from ...lib import utils

SETTINGS = utils.read_settings()
MAXLENGTH = 200  # number of words in the longest possible sentence. (longer sentences will be discarded)

train_raw_path = SETTINGS.get('paths', 'dataBncRawTrain')
train_parse_path = SETTINGS.get('paths', 'dataBncParsedTrain')
test_raw_path = SETTINGS.get('paths', 'dataBncRawTest')
test_parse_path = SETTINGS.get('paths', 'dataBncParsedTest')
heldout_raw_path = SETTINGS.get('paths', 'dataBncRawHeldout')
heldout_parse_path = SETTINGS.get('paths', 'dataBncParsedHeldout')

stanford_parser_dir = os.path.join(os.getcwd(), SETTINGS.get('paths', 'stanfordParser'))
my_path_to_jar = os.path.join(stanford_parser_dir, 'stanford-parser.jar')
my_path_to_models_jar = os.path.join(stanford_parser_dir, 'stanford-parser-3.6.0-models.jar')
eng_model_path = os.path.join(stanford_parser_dir, 'edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz')

parser = StanfordParser(
    model_path=eng_model_path,
    path_to_models_jar=my_path_to_models_jar,
    path_to_jar=my_path_to_jar,
    java_options='-mx5000m'
)
parser._classpath = tuple([j for j in parser._classpath] + [stanford_parser_dir + '/slf4j-api.jar', stanford_parser_dir + '/slf4j-simple.jar'])


for r, ds, fs in os.walk(heldout_raw_path):
    ds.sort()
    fs.sort()
    file_counter = 0
    already_parsed = os.listdir(heldout_parse_path)
    files = [f for f in fs if f[:1] in ('E', 'F', 'G') and f not in already_parsed]

    files_count = len(files)
    for f in files:
        file_counter += 1
        print f, file_counter/float(files_count)

        in_path = os.path.join(r, f)
        with codecs.open(in_path, 'r', 'utf-8') as fl:
            sents = [l for l in fl if len(l.split()) <= MAXLENGTH]

        trees = parser.raw_parse_sents(sents)

        out_path = os.path.join(heldout_parse_path, f)
        utils.create_dir_for_file(out_path)
        with codecs.open(out_path, 'w', 'utf-8') as fl:
            for t in trees:
                for t_ in t:
                    print >>fl, ' '.join(unicode(t_).split())
