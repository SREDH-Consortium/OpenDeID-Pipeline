import json
import logging
import os
from glob import glob

from . import tokenizer
from . import tokenizerfix


def preprocess(in_paths, dataname, out_dir, tagging_scheme, meta_prefix, sent_seg=True, ignores=[], ignore_O=False):
    # dataname defines the train, valid or test data
    # maybe have arguments to set which steps to apply/skip?
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)s %(levelname)s %(asctime)s: %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info('Started logger')
    logger.info('Preprocessing for %s set.' % dataname)

    logger.info('Importing %s tagger' % tagging_scheme)
    # tagger = importlib.import_module('taggers.' + tagging_scheme) # handle invalid tagging schemes?
    if tagging_scheme == 'BIO':
        from .taggers import BIO as tagger
    elif tagging_scheme == 'BIESO':
        from .taggers import BIESO as tagger
    else:
        raise ValueError('Invalid tagging scheme')

    # if the files have already been preprocessed, then we just use the preprocessed csv
    if os.path.exists(os.path.join(out_dir,'tagged_sentences_joined' + '_ignore_O' * ignore_O + '_'+ dataname +'.csv')):
        sent_joined = os.path.join(out_dir,'tagged_sentences_joined' + '_ignore_O' * ignore_O + '_'+ dataname +'.csv')
    else:
        logger.info('Tokenizing...')
        tokenized, sents = tokenizer.tokenize(in_paths, os.path.join(out_dir,'tokenized_' + dataname + '.txt'), meta_prefix,
                                             sent_seg, sents_path=os.path.join(out_dir, 'sentences_' + dataname + '.txt'), ignores=ignores)


        logger.info('Fixing...')
        fixed = tokenizerfix.fix(tokenized, os.path.join(out_dir, 'fixed_' + dataname + '.txt'), meta_prefix)

        logger.info('Tagging...')
        tagged = tagger.tag(fixed, os.path.join(out_dir, 'tagged_' + dataname + '.txt'), meta_prefix)

        logger.info('Creating Sentences...')
        sent_tagged = tokenizer.prepsentences(sents_path= sents,tagged_path= tagged,
                                              out_path=os.path.join(out_dir, 'tagged_sentences_'+ dataname +'.txt'), meta_prefix=meta_prefix)
        sent_joined = tokenizer.tidysentences(tagged_sentence_path =sent_tagged,
                                                    csv_path=os.path.join(out_dir, 'tagged_sentences_joined'+'_ignore_O'*ignore_O+ '_' + dataname +'.csv'),
                                                    ignore_O=ignore_O)


    logger.info('Finished')
    return sent_joined

if __name__ == '__main__':
    # read options from command line?
    config_path='config/config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

        common = config['common']
        meta_prefix = common['meta_prefix']
        window_size = common['window_size']
        tagging_scheme = common['tagging_scheme']
        sent_seg = common['sent_seg']
        # ignores = common['rules']

        train = config['train']
        in_dir_train = train['in_dir_train']
        in_dir_valid = train['in_dir_valid']
        out_dir = train['out_dir']
        ignores = train['ignores']
        ignore_O = train['ignore_O']

        # assumes training... when running from cmdline, specify one of arguments, train or predict?

    in_paths_train = glob(os.path.join(in_dir_train, '*.xml'))
    in_paths_valid = glob(os.path.join(in_dir_valid, '*.xml'))

    preprocess(in_paths_train, in_paths_valid, out_dir, tagging_scheme, meta_prefix, sent_seg, ignores, ignore_O)
