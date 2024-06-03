from preprocess import preprocess
import os
import json
from glob import glob
import time
from modeling.trainBERT import trainBERT

def run(in_paths_train, in_paths_valid,  out_dir, model_name, tagging_scheme, meta_prefix, labels_path, sent_seg=True, n_epoch=20, patience = 5,
        batch_size=32, segment=True, maxlen = 128, ignores=[], ignore_O=False):
    transformerslist  = {"Biobert": "dmis-lab/biobert-base-cased-v1.1",
                         "clinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
                         "dischargebert": "emilyalsentzer/Bio_Discharge_Summary_BERT",
                         "BERT": "bert-base-uncased"}
    transformer_name = transformerslist[model_name]
    sent_joined_train = preprocess.preprocess(in_paths_train, 'train', out_dir, tagging_scheme, meta_prefix, sent_seg, ignores, ignore_O)
    sent_joined_valid = preprocess.preprocess(in_paths_valid, 'valid', out_dir, tagging_scheme, meta_prefix, sent_seg, ignores, ignore_O)
    model_path = trainBERT(sent_joined_train, sent_joined_valid, labels_path, out_dir, model_name, transformer_name, segment,
                           maxlen, batch_size, n_epoch, patience)
    return model_path

if __name__ == '__main__':
    # load arguments from config or command line
    os.chdir('../2022-v1')
    print(os.getcwd())
    with open('config/BERTconfig.json') as config:
        config = json.loads(config.read())

        common = config['common']
        meta_prefix = common['meta_prefix']
        tagging_scheme = common['tagging_scheme']
        labels_path = common['labels_path']
        model_name = common['model_name']
        sent_seg = common['sent_seg'] # this gives whether we require to split the document to sentences, TRUE for BERT
        maxlen = common['maxlen']
        segment = common['segment'] # segment long sentences or truncate


        train_cfg = config['train']
        in_dir_train = train_cfg['in_dir_train']
        in_dir_valid = train_cfg['in_dir_valid']
        out_dir = train_cfg['out_dir']
        n_epoch = train_cfg['n_epoch']
        patience = train_cfg['patience']
        batch_size = train_cfg['batch_size']
        ignores = train_cfg['ignores']
        ignore_O = train_cfg['ignore_O']


    print(f"Experimentinfo: in_dir: {in_dir_train},  in_dir_valid: {in_dir_valid}, out_dir: {out_dir}.")
    in_paths_train = glob(os.path.join(in_dir_train, '*.xml'))
    in_paths_valid = glob(os.path.join(in_dir_valid, '*.xml'))
    print(f"Experimentinfo: Train size:{len(in_paths_train)}, Valid size:{len(in_paths_valid)}")
    # for model_name in ["Biobert","clinicalbert","dischargebert","BERT"]:
    print(f"Modelinfo: Transformer: {model_name},  Maxlen:{maxlen}, Batch size: {batch_size}.")
    start = time.time()
    if os.path.exists(out_dir) is False:
        os.mkdir(out_dir)
    run(in_paths_train, in_paths_valid, out_dir, model_name, tagging_scheme,  meta_prefix, labels_path, sent_seg, n_epoch,
        patience, batch_size,  segment, maxlen,  ignores, ignore_O)
    end = time.time()
    print("total training time: %s"%(end-start))
