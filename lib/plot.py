import numpy  # type: ignore
import pandas  # type: ignore
import matplotlib.pyplot as plot  # type: ignore
import os
import csv
import datetime
import numpy as np

from . import utils

FIG_OUTPUT_PATH = utils.read_settings().get('paths', 'figures')


def init_pgf_fig(mpl, scale):
    def figsize(scale):
        fig_width_pt = 412.56496                          # Get this from LaTeX using \the\textwidth
        inches_per_pt = 1.0/72.27                       # Convert pt to inch
        golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
        fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
        fig_height = fig_width*golden_mean              # height in inches
        fig_size = [fig_width, fig_height]
        return fig_size

    pgf_with_latex = {                      # setup matplotlib to use latex for output
        "pgf.texsystem": "xelatex",        # change this if using xetex or lautex
        "text.usetex": True,                # use LaTeX to write all text
        "font.family": "serif",
        "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
        "font.sans-serif": [],
        "font.monospace": [],
        "axes.labelsize": 10,               # LaTeX default is 10pt font.
        "text.fontsize": 10,
        "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.figsize": figsize(scale),     # default fig size of 0.9 textwidth
        "pgf.preamble": [
            r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
            r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
            ]
        }
    mpl.rcParams.update(pgf_with_latex)


def save_fig(plt, fname):
    plt.savefig(os.path.join(FIG_OUTPUT_PATH, fname) + '.pgf', bbox_inches='tight')
    plt.savefig(os.path.join(FIG_OUTPUT_PATH, fname) + '.pdf', bbox_inches='tight')


def plot_cm(cm, y_names):
    """
    Plots the confusion matrix

    :param cm:
    :return:
    """
    plot.figure()
    plot.imshow(cm, interpolation='nearest', cmap=plot.cm.Blues)
    plot.colorbar()
    tick_marks = numpy.arange(len(y_names))
    plot.xticks(tick_marks, y_names, rotation=45)
    plot.yticks(tick_marks, y_names)
    plot.tight_layout()
    plot.ylabel('True label')
    plot.xlabel('Predicted label')
    plot.show()


def log_result(model_type, features, model_call, train_df, test_df, result, result_train, data_source='penn', path=None):
    # type: (str, List[str], str, str, str, str) -> None
    if not path:
        settings = utils.read_settings()
        path = settings.get('paths', 'logModelResults')
    csv_file = os.path.join(path, data_source, model_type + '.csv')
    file_is_new = not os.path.isdir(path) or not os.path.isfile(csv_file)
    if file_is_new:
        utils.create_dir_for_file(csv_file)
        f = open(csv_file, 'w')
    else:
        f = open(csv_file, 'a')
    writer = csv.writer(f)
    if file_is_new:
        writer.writerow(['accuracy', 'accuracy-train', 'model', 'call', 'features', 'train data', 'test data', 'time'])
    writer.writerow([
        result, result_train, model_type, model_call, ' '.join(sorted(features)), train_df, test_df,
        datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
    ])


def create_result_df(y_predicted, test_df, save_path):
    test_df.loc[:, 'Y_PREDICTED'] = pandas.Series(y_predicted, index=test_df.index)
    test_df.to_pickle(save_path)


def export_errors(y_predicted, y_true, examples_df, save_path):
    # type: (List, List, pandas.DataFrame, str) -> None
    examples_df.loc[:, 'Y_PREDICTED'] = pandas.Series(y_predicted, index=examples_df.index)
    examples_df.loc[:, 'Y_TRUE'] = pandas.Series(y_true, index=examples_df.index)
    for pred in pandas.unique(y_predicted):
        for tru in pandas.unique(y_true):
            if pred != tru:
                path = os.path.join(save_path, pred + '-' + tru)
                utils.create_dir_for_file(path)
                selected = examples_df.loc[(examples_df['Y_PREDICTED'] == pred) & (examples_df['Y_TRUE'] == tru)]
                selected.to_csv(path + '.csv', sep='|')
                with open(path + '-sent.txt', 'w') as f:
                    for sent in selected.loc[:, '_sent']:
                        f.write(sent + "\n")


def get_accuracy(y_predicted, y_true):
    assert len(y_predicted) == len(y_true)
    return sum(y_predicted == y_true)/float(len(y_true))