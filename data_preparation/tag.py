import os
import os.path

from ..lib import corpus, utils


SETTINGS = utils.read_settings()


def tag_file(old_file, new_file):
    utils.create_dir_for_file(new_file)

    call = ' '.join([
        "java", "-classpath",
        os.path.join(
            SETTINGS.get('paths', 'tagger'),
            'mxpost.jar'
        ),
        "-mx30m",
        'tagger.TestTagger',
        os.path.join(
            SETTINGS.get('paths', 'tagger'),
            'tagger.project<'
        ),
        old_file,
        ">",
        new_file
    ])
    os.system(call)


if __name__ == '__main__':
    old_path = SETTINGS.get('paths', 'dataRaw')
    new_path = SETTINGS.get('paths', 'dataPOS')

    corpus.walk_and_transform(tag_file, old_path, new_path)
