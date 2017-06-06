from enum import Enum
import os
import sets
import utils

from .Tree import Tree


class DataType(Enum):
    ALL = 0
    TRAIN = 1
    HELDOUT = 2
    TEST = 3


def walk_and_transform(function, input_corpus_path, output_corpus_path):
    """
    Applies the function to all the files from the input corpus together with the
    corresponding files from the output corpus
    """
    for r, ds, fs in os.walk(input_corpus_path):
        print r
        ds.sort()
        fs.sort()
        for f in fs:
            old_file = os.path.join(r, f)
            new_file = os.path.join(output_corpus_path, os.path.basename(r), f)
            function(old_file, new_file)


def walk_parses(function, data_type=DataType.TRAIN):
    settings = utils.read_settings()
    path = settings.get('paths', 'dataParsed')

    leave_out_dirs = []
    if data_type == DataType.TRAIN:
        leave_out_dirs = [os.path.join(path, dir_nb) for dir_nb in ('22', '23', '24')]

    for r, ds, fs in os.walk(path):
        if r in leave_out_dirs:
            continue
        print r
        ds.sort()
        fs.sort()
        for f in fs:
            f_path = os.path.join(r, f)
            with open(f_path, 'r') as f:
                for l in f:
                    t = Tree.from_string(l)
                    function(t)



# WAITING FOR
# DELETION::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

def walk1(function, corpusFileType, result=None, data_type="train_devel"):
    path = _getCorpusFileTypePath(corpusFileType)
    path = os.path.join(path, data_type)

    dirnames = [
        dirname for dirname in os.listdir(path)
        if os.path.isdir(os.path.join(path, dirname))
    ]
    dirnames = sorted(dirnames)

    if data_type == "train_devel":
        # so this option means all the data? TODO redo to something reasonable
        pass
    elif data_type == "test_devel":
        dirnames = ['24']
    elif data_type == "test":
        dirnames = ['23']
    else:
        raise NameError("Unknown data type: one of the following accepted: train_devel, test_devel, test")

    dirnames = [os.path.join(path, d) for d in dirnames]
    for d in dirnames:
        print d
        for fname in sorted(os.listdir(d)):
            f = os.path.join(d, fname)
            result = function(f, result)

    return result

def walk2(function, corpusFileType, result=None, restrictToFiles=[], data_type="train_devel"):
    #TODO: spoj s funkci nahore
    '''
    tohle jsem pouzival pro pripravu dat, pro pruchod uz naparsovanejch vet pouzivam to nahore

    aplies the function to all the corpus files of the given type (orig, raw,
    parsed, ...); if the restrictToFiles argument is given, only the files
    corresponding to the provided number(s) will be searched

    @data_type: {test, test_devel, train_devel} - part of the data that should be considered
    '''
    path = _getCorpusFileTypePath(corpusFileType)
    if corpusFileType != 'orig' and corpusFileType != 'origP3':
        # files already divided into test, test_devel, train_devel
        paths = [os.path.join(path, dir) for dir in os.listdir(path)]
    else:
        paths = [path]

    restrictToDirs = []
    if restrictToFiles:
        if type(restrictToFiles) is str:
            restrictToFiles = [restrictToFiles]
        restrictToDirs = sets.Set([f[:2] for f in restrictToFiles])
        restrictToFiles = ['wsj_'+f for f in restrictToFiles]

    for path in paths:
        # get list of all the directories the corpus consists of:
        dirnames = [dirname for dirname in os.listdir(path)
            if os.path.isdir(os.path.join(path, dirname))]
        dirnames = sorted(dirnames)
        if '22' in dirnames:
            dirnames.remove('22')
        if '23' in dirnames:
            dirnames.remove('23')
        if '24' in dirnames:
            dirnames.remove('24')

        if restrictToDirs:
            dirnames = [dir for dir in dirnames if dir in restrictToDirs]
        # create the full paths, not just dir names:


        dirnames = [os.path.join(path, dir) for dir in dirnames]
        print dirnames

        for dir in dirnames:
            print dir
            for fname in sorted(os.listdir(dir)):
                print fname
                if restrictToFiles and fname not in restrictToFiles:
                    continue
                f = os.path.join(dir, fname)
                result = function(f, result)

    return result


def getSaveLocation(f, corpusFileType):
    '''for the given file and desired output corpusFileType, it returns the
    saving path for the file within the corpusFileType folder'''
    path = _getCorpusFileTypePath(corpusFileType)
    [parPath, fName] = os.path.split(f)
    par = os.path.split(parPath)[1]
    parAndFile = os.path.join(par, fName)
    parNb = int(par)

    settings = utils.readSettings()
    if parNb <= 21:
        path = os.path.join(path, settings.get('paths', 'trainDevelDir'),
                            parAndFile)
    elif parNb == 23:
        path = os.path.join(path, settings.get('paths', 'testDir'),
                            parAndFile)
    elif parNb == 24:
        path = os.path.join(path, settings.get('paths', 'testDevelDir'),
                            parAndFile)
    else:
        raise NameError("There should definitely be no directory like: " + dir)

    return path


def _getCorpusFileTypePath(corpusFileType):
    '''for the given corpusFileType (orig, pos, parsed, ...), it checks whether
    it is a valid type and returns its full path from the .settings file'''
    corpusFileTypeValues = ['orig', 'raw', 'pos', 'parsed', 'features']
    if corpusFileType not in corpusFileTypeValues:
        raise NameError(
            "undefined corpusFileType, use one of "+str(corpusFileTypeValues)
        )
    settings = utils.readSettings()
    path = settings.get('paths', 'data'+corpusFileType.capitalize())
    return path
