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
file_nb = len(f_names)
data = random.sample(f_names, file_nb)

# we split by 8 : 1 : 1 ratio:
train_cutoff_idx = int(file_nb * 0.8)
heldout_cutoff_idx = int(file_nb * 0.9)

train_data = data[:train_cutoff_idx]
heldout_data = data[train_cutoff_idx:heldout_cutoff_idx]
test_data = data[heldout_cutoff_idx:]


train_path = os.path.join(SETTINGS.get('paths', 'dataBncRawTrain'))
heldout_path = os.path.join(SETTINGS.get('paths', 'dataBncRawHeldout'))
test_path = os.path.join(SETTINGS.get('paths', 'dataBncRawTest'))

for data, path in zip((train_data, heldout_data, test_data), (train_path, heldout_path, test_path)):
    for f in data:
        dest_path = os.path.join(path, os.path.splitext(os.path.basename(f))[0])
        utils.create_dir_for_file(dest_path)
        copyfile(f, dest_path)
