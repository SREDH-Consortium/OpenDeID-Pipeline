import logging
import os
import json

def tag(tokens_path, predictions_path, out_path, meta_prefix):
    '''Combine tokens and predictions'''
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)s %(levelname)s %(asctime)s: %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info('Opening files')
    with open(predictions_path, 'r') as f:
        predictions = [line.strip() for line in f.readlines()]
    with open(tokens_path, 'r') as f:
        tokens = [line.strip() for line in f.readlines()]

    logger.info('Writing files')
    i = 0
    out_file = open(out_path, 'w')
    for line in tokens:
        if not line.startswith(meta_prefix):
            token, _, start = line.partition('\t')
            line = '%s\t%s\t%s' % (token, start, predictions[i])
            i += 1
        out_file.write(line + '\n')
    out_file.close()
    return out_path

if __name__ == '__main__':
    config_path='config/config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

        common = config['common']
        meta_prefix = common['meta_prefix'].encode('ascii')

        predict_cfg = config['predict']
        out_dir = predict_cfg['out_dir']

    predictions_path = os.path.join(out_dir, 'predicted_labels.txt')
    tokens_path = os.path.join(out_dir, 'fixed.txt')
    out_path = os.path.join(out_dir, 'predictions.txt')

    tag(tokens_path, predictions_path, out_path, meta_prefix)