from preprocess import preprocess
from postprocess import postprocess, postprocessingrules, combine
import os
import json
from glob import glob
import logging
import time
from modeling.predictBERT import predictBERT


def run(in_paths, out_dir, model_name, model_dir, tagging_scheme, labels_path, meta_prefix, dict_dir, dtd_path, transformer_name="emilyalsentzer/Bio_ClinicalBERT",
        segment=True, maxlen=128, batch_size = 32, rules=[], surrogate=False):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)s %(levelname)s %(asctime)s: %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    in_paths = [os.path.normpath(p) for p in in_paths]
    out_dir = os.path.normpath(out_dir)
    dict_dir = os.path.normpath(dict_dir)
    sent_joined_test = preprocess.preprocess(in_paths, 'test', out_dir, tagging_scheme, meta_prefix, sent_seg, rules, False)
    out_dir0 = out_dir
    out_dir = os.path.join(out_dir, model_name)
    if os.path.exists(out_dir) is False:
        os.mkdir(out_dir)

    model_path = os.path.join(model_dir, 'best_network_refit.pth')
    predictions = predictBERT(sent_joined_test, model_path, labels_path, os.path.join(out_dir, 'predicted_labels.txt'),
                              transformer_name, segment, maxlen, batch_size)

    combined = combine.tag(os.path.join(out_dir0, 'fixed_test.txt'), predictions, os.path.join(out_dir, 'predictions.txt'), meta_prefix)
    combined2 = postprocessingrules.tag(combined, os.path.join(out_dir, 'predictions2.txt'), meta_prefix, rules)
    postprocess.postprocess(tagging_scheme, combined2, os.path.join(out_dir, 'xml'), dtd_path, meta_prefix,
                            os.path.join(out_dir, 'mask'), dict_dir, surrogate)

if __name__ == '__main__':
    os.chdir('../2022-v1')
    with open('config/BERTconfig.json') as config:
        config = json.loads(config.read())

        common = config['common']
        meta_prefix = common['meta_prefix']
        tagging_scheme = common['tagging_scheme']
        labels_path = common['labels_path']
        dict_dir = common['dict_dir']
        model_name = common['model_name']
        dtd_path = common['dtd_path']
        sent_seg = common['sent_seg']
        maxlen = common['maxlen']
        segment = common['segment']

        predict_cfg = config['predict']
        in_dir = predict_cfg['in_dir']
        out_dir = predict_cfg['out_dir']
        rules = predict_cfg['rules']
        surrogate = predict_cfg['surrogate']
        batch_size = predict_cfg['batch_size']

        train_cfg = config['train']
        model_out_dir = train_cfg['out_dir']

    in_paths = glob(os.path.join(in_dir, '*.xml'))
    separate = False
    start = time.time()
    transformerslist ={"clinicalbert":"emilyalsentzer/Bio_ClinicalBERT",
                       "Biobert":"dmis-lab/biobert-base-cased-v1.1",
                       "dischargebert":"emilyalsentzer/Bio_Discharge_Summary_BERT",
                       "BERT": "bert-base-uncased"}

    # for model_name in ["clinicalbert","dischargebert","BERT"]:
    #    for maxlen in [128]:
    transformer_name = transformerslist[model_name]
    model_dir = os.path.join(model_out_dir, model_name)

    run(in_paths, out_dir, model_name, model_dir, tagging_scheme, labels_path, meta_prefix, dict_dir, dtd_path, transformer_name, segment,
        maxlen, batch_size, rules, surrogate)
    end = time.time()
    print("predict time: %s"%(end-start))