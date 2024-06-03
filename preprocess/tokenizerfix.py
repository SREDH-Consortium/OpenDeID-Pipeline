import json
import logging
import os
import re

def rules_fix(text):
    # match = re.match(r'(\d{2,})([A-Za-z]+)\n', line)
    # match2 = re.match(r'(\d+)([A-Za-z]{2,})\n', line)
    # r'(.+)/(.+)'
    # how deal with 'DuboisChief'?
    matches = [
        # jiaxing update:
        re.match(r'(.*\d)([,;])(\d.+)', text), # e.g. 14H10090,14N01358
        re.match(r'(.*[A-Za-z])([,;])([A-Za-z].+)', text), # e.g. HM;HM

        # original
        re.match(r'(.+\d)([.])?([A-Za-z].*)', text), # e.g. 123Doctor, 12.Man
        re.match(r'(.*\d)([.])?([A-Za-z].+)', text),
        re.match(r'(.+[A-Za-z])([.])?(\d.*)', text),
        re.match(r'(.*[A-Za-z])([.])?(\d.+)', text),

        re.match(r'(.*[a-z]{2})([A-Z][A-Za-z]+)', text) # e.g. DuboisChief, JoePsychiatric
    ]
    for match in matches:
        if match:
            fixed = []
            if match.start() > 0:
                fixed += [text[:match.start()]]
            fixed += [z for y in [rules_fix(x) for x in match.groups() if x] for z in y]
            if match.end() < len(text):
                fixed += [text[match.end():]]
            return fixed
    return [text]

def fix(in_path, out_path, meta_prefix):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)s %(levelname)s %(asctime)s: %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info('Started logger')

    logger.info('Reading %s', in_path)
    with open(in_path, 'r') as f:
        tagged = f.readlines()
    f = open(out_path, 'w')
    logger.info('Writing %s', out_path)
    for line in tagged:
        if line.startswith(meta_prefix + 'STARTDOC'):
            id_removed = False
        elif not id_removed and line.strip().partition('\t')[0][-1] == '|':
            id_removed = True
            continue
        f.write(line)
    f.close()
    return out_path

if __name__ == '__main__':
    # config or command line
    config_path='config/config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
        dir = os.path.join(config['out_dir'], 'preprocess')
        meta_prefix = config['meta_prefix'].encode('ascii')
        
    in_path = os.path.join(dir, 'tagged.txt')
    out_path = os.path.join(dir, 'fixed.txt')

    fix(in_path, out_path, meta_prefix)