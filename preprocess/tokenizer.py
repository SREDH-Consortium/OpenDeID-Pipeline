import errno
import json
import logging
import os
import re
from glob import glob
import pandas as pd
import spacy
from lxml import etree

from .tokenizerfix import rules_fix


#TODO provide brief descriptions
#TODO in preprocess you tokenize, segement sentences, obtain tokens in a window then vectorize

def fix(token_text, token_start, token_end, tags, meta_prefix):
    if not tags or int(tags[-1]['start']) > token_end or int(tags[-1]['end']) < token_start:
        return token_text + '\n'
    tag_type = tags[-1]['TYPE']
    tag_start = int(tags[-1]['start'])
    tag_end = int(tags[-1]['end'])
    index_start = tag_start - token_start
    index_end = tag_end - token_end - 1
    out = ''
    if tag_end <= token_end:
        if tag_start >= token_start: 
            before = token_text[:index_start]
            tag = token_text[index_start:index_end]
            after = token_text[index_end:]
            if before:
                out += '%s\n' % before
            out += '%sSTARTTAG\t%s\n' % (meta_prefix, tag_type)
            out += '%s\t%s\n' % (tag, token_start)
            out += '%sENDTAG\n' % meta_prefix
        elif tag_start < token_start: # else
            tag = token_text[:index_end]
            after = token_text[index_end:]
            out += '%s\t%s\n' % (tag, token_start)
            out += '%sENDTAG\n' % meta_prefix
        tags.pop()
        if after:
            out += fix(after, token_end - len(after) + 1, token_end, tags, meta_prefix)
    elif tag_end > token_end:
        if tag_start >= token_start:
            before = token_text[:index_start]
            tag = token_text[index_start:]
            if before:
                out += '%s\n' % before
            out += '%sSTARTTAG\t%s\n' % (meta_prefix, tag_type)
            out += '%s\t%s\n' % (tag, token_start)
        elif tag_start < token_start:
            tag = token_text
            out += '%s\t%s\n' % (tag, token_start)
    return out

def convert(token_text, token_start, tags, meta_prefix):
    out = ''
    token_end = token_start + len(token_text) - 1
    if tags and token_start >= int(tags[-1]['end']):
        tags.pop()
        out += '%sENDTAG\n' % meta_prefix
    if tags and not (int(tags[-1]['start']) > token_end or int(tags[-1]['end']) < token_start):
        tag_type = tags[-1]['TYPE']
        tag_start = int(tags[-1]['start'])
        tag_end = int(tags[-1]['end'])
        index_start = tag_start - token_start
        index_end = tag_end - token_end - 1
        out += fix(token_text, token_start, token_end, tags, meta_prefix)
    else:
        out += '%s\t%s\n' % (token_text, token_start)
    return out

def tokenize(in_paths, out_path, meta_prefix, sent_seg=True, sents_path=[],ignores=[]):
    """Tokenize given XML files, labelling tags as necessary.

    Args:
        in_paths: Paths to the XML files to be processed.
        out_path: The path to the file the output will be written to.
        meta_prefix: The prefix which indicates a line consists of metadata.
        sent_seg: whether to apply sentence segmentation or not
        sents_path: The path to the file that sentences will be written to.
        ignores: Tags that are not considered in the NER task.
    Returns:
        out_path: The path to the file the output was written to.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)s %(levelname)s %(asctime)s: %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info('Started logger')

    logger.info('Loading spacy')
        
    disabled_units = ['tagger', 'entity']
    if not sent_seg:
        disabled_units.append('parser')
    nlp = spacy.load('en_core_web_sm', disable=disabled_units)
    # disabling uneeded features for performance
    # whitespace/newlines are tokens

    logger.info('Opening output files')
    out_dir = os.path.dirname(out_path)
    if not os.path.exists(out_dir):
        try:
            os.makedirs(out_dir)
        except OSError as e:
            # check for race condition (i.e. if dirs were made in between)
            if e.errno != errno.EEXIST:
                raise # other error
    tagged = open(out_path, 'w', encoding='utf-8')
    if sent_seg:
        sents = open(sents_path, 'w')

    logger.info('Processing input files')
    # Currently assuming input are xml files
    for file_path in in_paths:
        file_path = os.path.normpath(file_path)
        logger.debug('Processing ' + file_path)
        if file_path.endswith('.xml'):
            tree = etree.parse(file_path)
            root = tree.getroot()
            text = root[0].text
            tags = [dict(tag.attrib) for tag in root[1]]
            tags.reverse()

            tags = [tag for tag in tags if tag['TYPE'] not in ignores]
        else:  # for predicting txt files, not used in train as there are no get tags
            text = open(file_path).read().strip()
            tags = []

        logger.debug('Tokenizing')
        doc = nlp(text, disable=['tagger']) # disabling uneeded features for performance
        # margin = 0
    
        # for sent in doc.sents: # if need to do sentence by sentence
        # TODO: note sentences so can remove sentences which just consist of O
        if sent_seg:
            sents.write('%sSTARTDOC\t%s\n' % (meta_prefix, file_path))
            for sent in doc.sents:
                sents.write(sent.text.strip() + '\n')

        logger.debug('Tagging')
        # new_tag = True
        tagged.write('%sSTARTDOC\t%s\n' % (meta_prefix, file_path))
        for i in range(len(doc)):
            # if doc[i].is_stop:  # removing stopwords, need to change tags' positions at the same time
            #     margin = len(doc[i].text) 
            #     continue

            token_text = doc[i].text.strip()
            # token_text = doc[i].lemma_.strip().encode('utf-8')  # use lemma. also need to change tags' positions since tokens' length changed
            if len(token_text) == 0:
                continue # skip whitespace tokens; maybe custom tokenizer would be better?
            start = doc[i].idx # + margin
            for token in rules_fix(token_text):
                tagged.write(convert(token, start, tags, meta_prefix))
                start += len(token) 
            
            # for predicting using rules
            if doc[i].like_url:
                pass
            elif doc[i].like_email:
                pass
        tagged.write('%sENDDOC\n' % meta_prefix)

    logger.info('Closing output files')
    tagged.close()
    if sent_seg:
        sents.close()
    return out_path, sents_path

def remove_o_sents(tagged_path, sents_path, out_path, out_sent_path, meta_prefix):
    with open(sents_path) as s, open(tagged_path) as t, open(out_path, 'w') as f, open(out_sent_path, 'w') as so:
        sents = s.readlines()
        tags = t.readlines()
        for sent in sents:
            if len(sent.strip()) == 0:
                continue
            if sent.startswith(meta_prefix):
                print(sent)
                f.write(sent)
                continue
            tagged_sent = []
            all_O = True

            while len(tags) > 0:
                tagged = tags.pop(0).strip()
                if tagged.startswith(meta_prefix):
                    continue
                token, _, tag = tagged.partition('\t')
                tagged_sent.append(tagged)
                if tag != 'O':
                    all_O = False
                if ''.join([token_pair.split('\t')[0] for token_pair in tagged_sent]).strip() == re.sub(r"[\n\t\s]", "", sent):
                    if not all_O:
                        f.write('\n'.join(tagged_sent) + '\n')
                        so.write(sent+'\n')
                    break
        return out_path


### updated by jiaxing 2022
#sents_path = 'D:\\Python\\TF\\OpenID\\4. OpenDeID Corpus and OpenDeID pipeline - Zoie Tokyo\\3. UNSW OpenDeID Pipeline\\2022\\output\\setting1\\train\\sentences.txt'
#tagged_path = 'D:\\Python\\TF\\OpenID\\4. OpenDeID Corpus and OpenDeID pipeline - Zoie Tokyo\\3. UNSW OpenDeID Pipeline\\2022\\output\\setting1\\train\\tagged.txt'
#out_path = 'D:\\Python\\TF\\OpenID\\4. OpenDeID Corpus and OpenDeID pipeline - Zoie Tokyo\\3. UNSW OpenDeID Pipeline\\2022\\output\\setting1\\train\\tagged_sentences.txt'
#meta_prefix = "#$"
#out_path = prepsentences(sents_path, tagged_path, out_path, meta_prefix)
def prepsentences(sents_path, tagged_path, out_path, meta_prefix):
    with open(sents_path) as s, open(tagged_path) as t, open(out_path, 'w') as f:
        sents = s.readlines()
        tags = t.readlines()
        sentID = 0
        for sent in sents:
            if len(sent.strip()) == 0:
                continue
            if sent.startswith(meta_prefix):
                fileID = sent.split('\t')[1].split('\n')[0]
                continue
            tagged_sent = []
            while len(tags) > 0:
                tagged = tags.pop(0).strip()
                if tagged.startswith(meta_prefix):
                    continue
                tagged = tagged+'\t'+str(sentID)+'\t'+fileID
                tagged_sent.append(tagged)
                if ''.join([token_pair.split('\t')[0] for token_pair in tagged_sent]).strip() == re.sub(r"[\n\t\s]","", sent):
                    f.write('\n'.join(tagged_sent)+'\n')
                    sentID += 1
                    break
    return out_path


def tidysentences(tagged_sentence_path, csv_path, ignore_O):
    with open(tagged_sentence_path, 'r') as f:
        tags = f.readlines()
        entities = []
        for tagged in tags:
            entity = {}
            word, tag, sentID, documentID = tagged.split('\t')
            documentID = documentID.split('\n')[0]
            entity['word'] = word
            entity['tag'] = tag
            entity['sentID'] = sentID
            entity['documentID'] = documentID
            entities.append(entity)

        data = pd.DataFrame(entities, columns=['word', 'tag', 'sentID',  'documentID'])
        # let's create a new column called "sentence" which groups the words by sentence
        data['text'] = data[['sentID','word', 'tag']].groupby(['sentID'])['word'].transform(
            lambda x: ' '.join(x.astype(str)))
        # let's also create a new column called "word_labels" which groups the tags by sentence
        data['tags'] = data[['sentID','word', 'tag']].groupby(['sentID'])['tag'].transform(
            lambda x: ','.join(x.astype(str)))
        data = data[["text", "tags", "sentID", 'documentID']].drop_duplicates().reset_index(drop=True)

        if ignore_O:
            data['length'] = data.text.apply(lambda x: len(x.split()))
            data['O_nums'] = data.tags.apply(lambda x: len([tag for tag in x.split(',') if tag == 'O']))
            data = data[["text", "tags", "sentID", 'documentID']][(data['length'] - data['O_nums']) != 0].reset_index(drop=True)

        data.to_csv(csv_path)
    return csv_path


if __name__ == '__main__':
    # could read options from command line to pass in
    config_path='config/config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

        common = config['common']
        meta_prefix = common['meta_prefix'].encode('ascii')
        # ignores = common['rules']
        sent_seg = common['sent_seg']

        train = config['train']
        in_dir = train['in_dir']
        out_dir = train['out_dir']
        ignores = train['ignores']

        # assumes training... when running from cmdline, specify one of arguments, train or predict?

    out_path = os.path.join(out_dir, 'tokenized.txt')
    sents_path = os.path.join(out_dir, 'sentences.txt')
    in_paths = glob(os.path.join(in_dir, '*.xml'))

    tokenize(in_paths, out_path, meta_prefix, sent_seg, sents_path, ignores)