import re
import logging
import os
import json

def tag_token(text, rules):
    if 'DATE' in rules:
        matches = [
            re.match(r'\d+/\d+/\d+', text), # 1/12/2000
            re.match(r'\d+-\d+-\d+', text), # 1-12-2000
            re.match(r'[A-Za-z]+/\d+/\d+', text), # Jan/12/2000
            re.match(r'\d+/[A-Za-z]+/\d+', text), # 12/Jan/2000
            re.match(r'[A-Za-z]+-\d+-\d+', text), # Jan-12-2000
            re.match(r'\d+-[A-Za-z]+-\d+', text), # 12-Jan-2000
        ]
        if any(matches):
            return 'B_DATE'
    if 'PHONE' in rules:
        matches = [
            re.match(r'\d{10}', text), # 8083131345
            re.match(r'\d{3}\.\d{3}\.\d{4}', text), # 808.313.1345
            re.match(r'\d{3}-\d{3}-\d{4}', text), # 808-313-1345, problem since is separated into multiple tokens
        ]
        if any(matches):
            return 'B_PHONE'
    if 'URL' in rules:
        matches = [
            re.match(r'(https?://)?(\w+\.)?\w+\.\w+(\.\w+)(/\w*)*/?', text),
        ]
        if any(matches):
            return 'B_URL'
    return None

def tag(predictions_path, out_path, meta_prefix, rules=[]):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)s %(levelname)s %(asctime)s: %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info('Applying rules...')
    predictions = open(predictions_path, 'r')
    out_file = open(out_path, 'w')
    for line in predictions:
        line = line.strip()
        if not line.startswith(meta_prefix):
            token, start, tag = line.split('\t')
            if tag == 'O': # prioritise nn over rules
                tag = tag_token(token, rules) or tag
            line = '%s\t%s\t%s' % (token, start, tag)
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
        rules = predict_cfg['rules']

    predictions_path = os.path.join(out_dir, 'predictions.txt')
    out_path = os.path.join(out_dir, 'predictions2.txt')

    tag(predictions_path, out_path, meta_prefix, rules)