import os
import logging
import pandas as pd
import json
from transformers import BertForTokenClassification,AutoModelForTokenClassification
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from torch import cuda
from tqdm import tqdm

from .dataloader import dataset
from .sequence import segment_sequence

def predictBERT(tagged_sent_csv, model_path, labels_path, out_path, transformer_name, segment=False, maxlen=128, batch_size = 32):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)s %(levelname)s %(asctime)s: %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info("Started Logger")

    logger.info('Loading O-Tagged Sentences (data)')
    testdata = pd.read_csv(tagged_sent_csv).drop(columns=['Unnamed: 0'])
    testdata['text'] = testdata['text'].astype(str)
    testdata['tags'] = testdata['tags'].astype(str)

    if segment:
        testdata = segment_sequence(origdata=testdata, maxlen=maxlen)

    # load the tags
    with open(labels_path, "r") as f:
        labels = f.readlines()
        labels = [label.split('\n')[0] for label in labels]
    labels_to_ids = {k: v for v, k in enumerate(labels)}
    ids_to_labels = {v: k for v, k in enumerate(labels)}

    logger.info('Tokenizing Sentences (data)')
    tokenizer = AutoTokenizer.from_pretrained(transformer_name, add_special_tokens=False,  add_prefix_space=True)

    testing_set = dataset(testdata, tokenizer, maxlen, labels_to_ids=labels_to_ids)
    test_params = {'batch_size': batch_size, 'shuffle': False, 'num_workers': 0}
    testing_loader = DataLoader(testing_set, **test_params)

    logger.info('Loading model from %s', model_path)
    device = 'cuda' if cuda.is_available() else 'cpu'
    logger.info('Device: %s' %device)

    #model = AutoModelForTokenClassification.from_pretrained(transformer_name, num_labels=len(labels_to_ids))
    #model.load_state_dict(torch.load(model_path))
    model = torch.load(model_path)
    model.to(device)

    test_labels, test_preds = test(model, testing_loader, device, ids_to_labels)
    predicted_labels = padpred(test_labels, test_preds, testdata)

    logger.info('Writing prediction to %s', out_path)
    with open(out_path, 'w') as f:
        for i in predicted_labels:
            f.write('%s\n' % i)
    return out_path

def padpred(test_labels, test_preds, testdata):
    assert len(test_preds) == testdata.shape[0]
    # pad 'O'
    for k in range(len(test_preds)):
        diff = len(testdata.tags[k].split(',')) - len(test_preds[k])
        if diff != 0:
            test_preds[k] = test_preds[k]+['O']*diff

    predicted_labels = [l for ls in test_preds for l in ls]
    return predicted_labels



def test(model, testing_loader, device, ids_to_labels):
    # put model in evaluation mode
    model.eval()
    test_labels, test_preds = [], []
    with torch.no_grad():
        with tqdm(total=len(testing_loader)) as t:
            for idx, batch in enumerate(testing_loader):
                ids = batch['input_ids'].to(device, dtype=torch.long)
                mask = batch['attention_mask'].to(device, dtype=torch.long)
                labels = batch['labels'].to(device, dtype=torch.long)

                outputs = model(input_ids=ids, attention_mask=mask)
                eval_logits = outputs[0]
                # eval_probs = F.softmax(eval_logits, dim=2)
                predictions = torch.argmax(eval_logits, axis=2)

                for s in range(labels.shape[0]):
                    label = torch.masked_select(labels[s], labels[s] != -100)
                    pred = torch.masked_select(predictions[s], labels[s] != -100)
                    label_id = [ids_to_labels[id] for id in label.detach().cpu().tolist()]
                    pred_id = [ids_to_labels[id] for id in pred.detach().cpu().tolist()]
                    test_labels.append(label_id)
                    test_preds.append(pred_id)

                t.set_description(desc="Testing on the independent testing set")
                t.set_postfix(steps=(idx + 1))
                t.update(1)

    return test_labels, test_preds


if __name__ == '__main__':
    config_path = '../config/BERTconfig.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

        common = config['common']
        labels_path = common['labels_path']
        model_path = common['model_path']
        # rules = common['rules']
        maxlen = common['maxlen']
        transformer_name = common['transformer_name']
        segment = common['segment']

        use_ortho = common['use_ortho']
        use_embedding = common['use_embedding']
        use_embedding_ehr = common['use_embedding_ehr']
        use_dict = common['use_dict']

        predict_cfg = config['predict']
        out_dir = predict_cfg['out_dir']
        x_path = predict_cfg['x_path']
        x1_path = predict_cfg['x1_path']
        orth_path = predict_cfg['orth_path']
        dict_path = predict_cfg['dict_path']
        batch_size = predict_cfg['batch_size']

    tagged_sent_csv = os.path.join(out_dir, 'tagged_sentences_joined.csv')
    out_path = os.path.join(out_dir, 'predicted_labels.txt')

    predictBERT(tagged_sent_csv, model_path, labels_path, out_path, transformer_name, segment, maxlen, batch_size)