import os
import random

from ...lib import utils
from shutil import copyfile


SETTINGS = utils.read_settings()


f_names = []
for r, ds, fs in os.walk(SETTINGS.get('paths', 'dataBncRaw')):
    ds.sort()
    fs.sort()
    for f in fs:
        f_names.append(os.path.join(r, f))

random.seed(45)
manual_test = random.sample(f_names, 8)
manual_test_fill_in = manual_test[:4]
manual_test_proofread = manual_test[4:]

dest_path_fill_in = os.path.join(SETTINGS.get('paths', 'dataBncTestManual'), 'fill_in')
dest_path_proofread = os.path.join(SETTINGS.get('paths', 'dataBncTestManual'), 'proofread')

for f in manual_test_fill_in:
    dest_path = os.path.join(dest_path_fill_in, os.path.basename(f))
    utils.create_dir_for_file(dest_path)
    copyfile(f, dest_path)

for f in manual_test_proofread:
    dest_path = os.path.join(dest_path_proofread, os.path.basename(f))
    utils.create_dir_for_file(dest_path)
    copyfile(f, dest_path)

