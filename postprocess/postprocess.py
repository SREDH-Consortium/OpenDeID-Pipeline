import os
import sys
import logging
from lxml import etree
import json
from collections import OrderedDict
from glob import glob
import random
import uuid
from .surrogate import Surrogator


def getElement(tag, dtd):
    # find appropriate tag name
    for el in dtd.iterelements():
        for attr in el.iterattributes():
            if attr.name == 'TYPE':
                if tag in attr.values():
                    return attr.elemname

def postprocess(scheme, predictions_path, out_dir, dtd_path, meta_prefix, mask_dir, dict_dir, surrogate=False):
    surrogator = None
    if surrogate:
        surrogator = Surrogator()
    if scheme == 'BIO':
        return BIO(predictions_path, out_dir, dtd_path, meta_prefix, mask_dir, surrogator)
    elif scheme == 'BIESO':
        return BIESO(predictions_path, out_dir, dtd_path, meta_prefix, mask_dir, surrogator)

def BIO(predictions_path, out_dir, dtd_path, meta_prefix, mask_dir, surrogator):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)s %(levelname)s %(asctime)s: %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info('Started logger')

    logger.info('Opening files')
    with open(predictions_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
        predictions = [line.rpartition('\t')[2] for line in lines if not line.startswith(meta_prefix)]
        tokens = [line.rpartition('\t')[0] if not line.startswith(meta_prefix) else line for line in lines]
    dtd = etree.DTD(dtd_path)
    tags = None
    tree = None
    in_path = None
    i = 0
    id = 0
    tag_start = None
    tag_text = ''
    tag = None
    attrib = OrderedDict()
    attrib['id'] = None
    attrib['start'] = None
    attrib['end'] = None
    attrib['text'] = None
    attrib['TYPE'] = None
    attrib['comment'] = None
    logger.info('Writing tags')
    for line in tokens:
        if line.startswith(meta_prefix):
            if line.startswith(meta_prefix + 'STARTDOC'):
                _, _, in_path = line.partition('\t')
                parser = etree.XMLParser(strip_cdata=False)
                tree = etree.parse(in_path, parser=parser)
                masked_text = tree.find('TEXT').text
                root = tree.getroot()
                tags = root[1]
                tags.text = '\n'
                id = 0
            elif line.startswith(meta_prefix + 'ENDDOC'):
                
                #if not dtd.validate(tree):
                #    logger.error('%s is invalid', in_file)
                #    logger.error(dtd.error_log.filter_from_errors()[0])
                # need to make dir if doesnt exist
                if not os.path.exists(out_dir):
                    try:
                        os.makedirs(out_dir)
                    except OSError as e:
                        # check for race condition (i.e. if dirs were made in between)
                        if e.errno != errno.EEXIST:
                            raise # other error
                if not os.path.exists(mask_dir):
                    try:
                        os.makedirs(mask_dir)
                    except OSError as e:
                        # check for race condition (i.e. if dirs were made in between)
                        if e.errno != errno.EEXIST:
                            raise # other error

                tree.write(os.path.join(out_dir, os.path.basename(in_path)), encoding='utf-8', pretty_print=True, xml_declaration=True)
                with open(os.path.join(mask_dir, os.path.basename(in_path.replace('xml', 'txt'))), 'w') as masked:
                    masked_text.append(original_text[int(attrib['end']):])
                    masked.write(''.join(masked_text))
            continue
        token, _, start = line.partition('\t')
        start = int(start)
        if tag:
            if predictions[i] == 'I_' + tag and tag_start + len(tag_text) + 1 == start:
                # if there is a tag, and the current token either is  the same tag and starts immediately after
                tag_text += ' ' + token
            elif predictions[i] == 'I_' + tag and tag_start + len(tag_text) == start:
                tag_text += token
            else: # write out
                attrib['id'] = 'P'+str(id)
                attrib['start'] = str(tag_start)

                last_pos = int(attrib['end']) if attrib['end'] is not None and int(attrib['end']) < tag_start else 0
                masked_text.append(original_text[last_pos: tag_start])

                attrib['end'] = str(tag_start+len(tag_text))
                attrib['text'] = tag_text
                attrib['TYPE'] = tag
                attrib['comment'] = ''
                masked_text.append("[%s]"%tag.upper())
                tag_el = etree.SubElement(tags, getElement(tag, dtd), attrib=attrib)
                tag_el.tail = '\n'
                # if not dtd.validate(tags[-1]):
                    # logger.error('%s is invalid', in_file)
                    # logger.error(dtd.error_log.filter_from_errors()[0])
                id += 1
                tag = None
                tag_text = ''
                tag_start = None
        if not tag and predictions[i].startswith('B_'):
            tag = predictions[i][2:]
            tag_start = int(start)
            tag_text = token
        i += 1
    logger.info('Finished')
    return out_dir

def BIESO(predictions_path, out_dir, dtd_path, meta_prefix, mask_dir, surrogator):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)s %(levelname)s %(asctime)s: %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info('Started logger')

    logger.info('Opening files')
    with open(predictions_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
        predictions = [line.rpartition('\t')[2] for line in lines if not line.startswith(meta_prefix)]
        tokens = [line.rpartition('\t')[0] if not line.startswith(meta_prefix) else line for line in lines]
    dtd = etree.DTD(dtd_path)
    tags = None
    tree = None
    in_path = None
    i = 0
    id = 0
    tag_start = None
    tag_text = ''
    tag = None
    attrib = OrderedDict()
    attrib['id'] = None
    attrib['start'] = None
    attrib['end'] = None
    attrib['text'] = None
    attrib['TYPE'] = None
    attrib['comment'] = None
    logger.info('Writing tags')
    new_start = 0
    line = tokens[0]
    for line in tokens:
        if line.startswith(meta_prefix):
            if line.startswith(meta_prefix + 'STARTDOC'):
                _, _, in_path = line.partition('\t')
                parser = etree.XMLParser(strip_cdata=False)
                tree = etree.parse(in_path, parser=parser)
                original_text = tree.find('TEXT').text
                masked_text = []
                root = tree.getroot()
                tags = root[1]
                tags.text = '\n'
                id = 0
            elif line.startswith(meta_prefix + 'ENDDOC'):
                #if not dtd.validate(tree):
                #    logger.error('%s is invalid', in_file)
                #    logger.error(dtd.error_log.filter_from_errors()[0])
                # need to make dir if doesnt exist
                if not os.path.exists(out_dir):
                    try:
                        os.makedirs(out_dir)
                    except OSError as e:
                        # check for race condition (i.e. if dirs were made in between)
                        if e.errno != errno.EEXIST:
                            raise # other error
                if not os.path.exists(mask_dir):
                    try:
                        os.makedirs(mask_dir)
                    except OSError as e:
                        # check for race condition (i.e. if dirs were made in between)
                        if e.errno != errno.EEXIST:
                            raise # other error
                

                tree.write(os.path.join(out_dir, os.path.basename(in_path)), encoding='utf-8', pretty_print=True, xml_declaration=True)
                with open(os.path.join(mask_dir, os.path.basename(in_path.replace('xml', 'txt'))), 'w') as masked:
                    masked_text.append(original_text[int(attrib['end']):])
                    masked.write(''.join(masked_text))

            continue
        #print(line)
        token, _, start = line.partition('\t')
        if start == '':
            start = new_start
        start = int(start)
        prediction = predictions[i]
        new_start = start
        if tag:
            if tag_start + len(tag_text) + 1 == start or tag_start + len(tag_text) == start:
                if tag_start + len(tag_text) + 1 == start:
                    tag_text += ' ' + token
                else:
                    tag_text += token
                if prediction == 'E_' + tag:
                    # end of tag
                    attrib['id'] = 'P'+str(id)
                    attrib['start'] = str(tag_start)
                    last_pos = int(attrib['end']) if attrib['end'] is not None and int(attrib['end']) < tag_start else 0
                    masked_text.append(original_text[last_pos: tag_start])
                    attrib['end'] = str(tag_start+len(tag_text))
                    attrib['text'] = tag_text
                    attrib['TYPE'] = tag
                    attrib['comment'] = ''
                    masked_text.append("[%s]"%tag.upper())

                    tag_el = etree.SubElement(tags, getElement(tag, dtd), attrib=attrib)
                    tag_el.tail = '\n'
                    # if not dtd.validate(tags[-1]):
                        # logger.error('%s is invalid', in_file)
                        # logger.error(dtd.error_log.filter_from_errors()[0])
                    id += 1
                    tag = None
                    tag_text = ''
                    tag_start = None
                elif not prediction == 'I_' + tag:
                    # malformed, clear tag
                    tag = None
                    tag_text = ''
                    tag_start = None
            else:
                # malformed, clear tag
                tag = None
                tag_text = ''
                tag_start = None
        if prediction.startswith('S'):
            # single tag, add immediately
            tag = prediction[2:]
            tag_start = int(start)
            tag_text = token
            attrib['id'] = 'P'+str(id)
            attrib['start'] = str(tag_start)
            last_pos = int(attrib['end']) if attrib['end'] is not None and int(attrib['end']) < tag_start else 0
            masked_text.append(original_text[last_pos: tag_start])
            masked_text.append("[%s]"%tag.upper())
            attrib['end'] = str(tag_start+len(tag_text))
            attrib['text'] = tag_text
            attrib['TYPE'] = tag
            attrib['comment'] = ''
            tag_el = etree.SubElement(tags, getElement(tag, dtd), attrib=attrib)
            tag_el.tail = '\n'
            id += 1
            tag = None
            tag_text = ''
            tag_start = None
        elif prediction.startswith('B'):
            # start of a new tag
            tag = predictions[i][2:]
            tag_start = int(start)
            tag_text = token
        i += 1
    logger.info('Finished')
    if surrogator is not None:
        logger.info('Genertating surrogate')
        import re
        surrogate_dir = re.sub(r'[\\/]xml([\\/]?)$', r'/surrogate/', out_dir)
        surrogator.multi_file_surrogate(out_dir, surrogate_dir)
        logger.info('Done')
    return out_dir


if __name__ == '__main__':
    with open('config/config.json') as config:

        config = json.load(config)

        common = config['common']
        meta_prefix = common['meta_prefix']
        tagging_scheme = common['tagging_scheme']
        dtd_path = common['dtd_path']
        dict_dir = common['dict_dir']

        predict_cfg = config['predict']
        out_dir = predict_cfg['out_dir']
        surrogate = predict_cfg['surrogate']

        predictions_path = os.path.join(out_dir, 'predictions2.txt')
        xml_dir = os.path.join(out_dir, 'xml')
        mask_dir = os.path.join(out_dir, 'mask')


    postprocess(tagging_scheme, predictions_path, xml_dir, dtd_path, meta_prefix, mask_dir, dict_dir, surrogate=True)