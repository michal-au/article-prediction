import csv
import datetime
import os
import subprocess


from ...lib.utils import read_settings
SETTINGS = read_settings()
KENLM = '$HOME/kenlm/build/bin'


def evaluate_by_perplexity(model_path, reference_data):
    call = ' '.join([
        os.path.join(KENLM, 'query'),
        '-v summary',
        model_path,
        '<', reference_data,
    ])
    print '... evaluating kenlm'
    output = subprocess.check_output(call, shell=True, stderr=subprocess.STDOUT)
    print '... done evaluating kenlm'
    perplex, perplex_without_oovs = [float(l.split('\t')[1]) for l in output.split('\n') if l.startswith('Perplexity')]
    return perplex, perplex_without_oovs


def log_perplexity(writer, model_name, perplexity, perplexity_without_oovs, reference_data):
    writer.writerow([
        model_name,
        perplexity,
        perplexity_without_oovs,
        reference_data,
        datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
    ])


# if __name__ == "__main__":
#
#     train_data = os.path.join(SETTINGS.get('paths', 'dataRnnlmRaw'), 'train')
#     heldout_data = os.path.join(SETTINGS.get('paths', 'dataRnnlmRaw'), 'heldout')
#     eval_output = os.path.join(SETTINGS.get('paths', 'logKennlmResults'), 'penn.csv')
#     writer = csv.writer(open(eval_output, 'a'))
#
#     for order in (3, 4, 5, 6, 7, 8, 9, 10):
#         model_name = 'kenlm-ptb-{}'.format(order)
#         model_path = os.path.join(SETTINGS.get('paths', 'modelLM'), model_name)
#
#         print 'training {}'.format(model_name)
#         train_kenlm(model_path, train_data, {'order': order})
#
#         print 'evaluating {}'.format(model_name)
#         perplx, perplx_without_oovs = evaluate_by_perplexity(model_path, heldout_data)
#         log_perplexity(
#             writer=writer,
#             model_name=model_name,
#             perplexity=perplx,
#             perplexity_without_oovs=perplx_without_oovs,
#             reference_data=heldout_data,
#         )
#
#     print 'DONE'

if __name__ == "__main__":
    model_type_id = 'lc-nbs'
    order = 5
    eval_data = os.path.join(SETTINGS.get('paths', 'dataBenchmarkPreprocessed'), 'train-{}'.format(model_type_id))
    eval_output = os.path.join(SETTINGS.get('paths', 'logKennlmResults'), 'ggl.csv')
    model_name = 'kenlm-ggl-{}-{}'.format(order, model_type_id)
    model_path = os.path.join(SETTINGS.get('paths', 'modelLM'), model_name)

    writer = csv.writer(open(eval_output, 'a'))
    perplx, perplx_without_oovs = evaluate_by_perplexity(model_path, eval_data)
    log_perplexity(
        writer=writer,
        model_name=model_name,
        perplexity=perplx,
        perplexity_without_oovs=perplx_without_oovs,
        reference_data=eval_data,
    )

    print 'DONE'
