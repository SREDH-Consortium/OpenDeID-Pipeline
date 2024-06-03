import json
from transformers import AutoModelForTokenClassification
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd
from torch import cuda
from tqdm import tqdm
import os
from .dataloader import dataset
from .sequence import segment_sequence

def trainBERT(sent_joined_train, sent_joined_valid, labels_path, out_dir, model_name, transformer_name, segment=False,
            maxlen = 128, batch_size=32, n_epoch=20, patience = 10):
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)s %(levelname)s %(asctime)s: %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info('Loading Tagged Train Sentences (data)')
    traindata = pd.read_csv(sent_joined_train).drop(columns=['Unnamed: 0'])
    traindata['text'] = traindata['text'].astype(str)
    traindata['tags'] = traindata['tags'].astype(str)

    logger.info('Loading Tagged Valid Sentences (data)')
    validdata = pd.read_csv(sent_joined_valid).drop(columns=['Unnamed: 0'])
    validdata['text'] = validdata['text'].astype(str)
    validdata['tags'] = validdata['tags'].astype(str)

    # segment the sentences exceed maxlen into two or more segments. if False, then it will be truncated.
    if segment:
        logger.info('Segmenting Long sentences')
        traindata = segment_sequence(origdata=traindata, maxlen=maxlen)
        validdata = segment_sequence(origdata=validdata, maxlen=maxlen)

    logger.info('Combining Train and Valid Sentences (data)')
    alldata = pd.concat([traindata, validdata]).reset_index(drop=True)


    # load the tags
    with open(labels_path, "r") as f:
        labels = f.readlines()
        labels = [label.split('\n')[0] for label in labels]
    labels_to_ids = {k: v for v, k in enumerate(labels)}
    #ids_to_labels = {v: k for v, k in enumerate(labels)}

    logger.info('Preparing the dataset and dataloader')
    train_params = {'batch_size': batch_size, 'shuffle': False, 'num_workers': 0}
    validation_params = {'batch_size': batch_size*2, 'shuffle': False, 'num_workers': 0}

    logger.info('Tokenization using  %s' %transformer_name)

    tokenizer = AutoTokenizer.from_pretrained(transformer_name, add_special_tokens=False,  add_prefix_space=True)

    training_set = dataset(traindata, tokenizer, maxlen, labels_to_ids)
    validation_set = dataset(validdata, tokenizer, maxlen, labels_to_ids)
    all_set = dataset(alldata, tokenizer, maxlen, labels_to_ids)

    training_loader = DataLoader(training_set, **train_params)
    validation_loader = DataLoader(validation_set, **validation_params)
    all_loader = DataLoader(all_set, **train_params)

    logger.info('Train BERT model')
    device = 'cuda' if cuda.is_available() else 'cpu'
    logger.info('Device: %s' %device)

    LEARNING_RATE = 2e-05
    MAX_GRAD_NORM = 10
    model = AutoModelForTokenClassification.from_pretrained(transformer_name, num_labels=len(labels_to_ids))
    model.to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)

    # the final model_dir is like: ./output/train_with_sentences/Biobert/
    model_dir = os.path.join(out_dir, model_name)
    if os.path.exists(model_dir) is False:
        os.mkdir(model_dir)

    model, outputs = train_model(model=model, training_loader=training_loader, validation_loader=validation_loader, all_loader=all_loader,
                                 save_dir=model_dir, device=device, optimizer = optimizer,  patience=patience, n_epochs=n_epoch,
                                 MAX_GRAD_NORM=MAX_GRAD_NORM)

    logger.info('Train BERT model Completed! Saving the loss and accuracy in train and validation.')
    train_losses, train_acces, train_f1es, valid_losses, valid_acces,  valid_f1es = outputs
    results = pd.DataFrame(zip(train_losses, valid_losses, train_acces,valid_acces, train_f1es, valid_f1es),
                           columns=['train_loss','valid_loss','train_accuracy','valid_accuracy', 'train_f1','valid_f1'])
    results.to_csv(model_dir+'/train_valid_results.csv', index=False)
    plot_model(results,model_dir=model_dir)
    return model_dir



class EarlyStopping:
    """Early stops the training if validation accuracy doesn't improve after a given patience."""
    def __init__(self, save_dir, patience=5, verbose=False, delta=0):
        """
        Args:
            save_dir : folder to save the model
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_dir = save_dir
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_accuracy_max = np.Inf
        self.delta = delta

    def __call__(self, val_acc, model):

        score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
            self.counter = 0

    def save_checkpoint(self, val_acc, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation accuracy increased ({self.val_accuracy_max:.6f} --> {val_acc:.6f}).  Saving model ...')
        path = os.path.join(self.save_dir, 'best_network.pth')
        torch.save(model.state_dict(), path)	# save the up-to-date best model
        self.val_accuracy_max = val_acc



def train_model(model,training_loader, validation_loader, all_loader, save_dir, device, optimizer,  patience=5, n_epochs=20, MAX_GRAD_NORM=10):
    train_losses = []
    train_acces = []
    train_f1es = []
    # 用数组保存每一轮迭代中，在测试数据上测试的损失值和精确度，也是为了通过画图展示出来。
    valid_losses = []
    valid_acces = []
    valid_f1es = []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(save_dir = save_dir, patience=patience, verbose=True)

    for epoch in range(1, n_epochs + 1):

        ###################
        # train the model #
        ###################
        tr_loss, tr_accuracy, tr_f1 = 0, 0, 0
        nb_tr_examples, nb_tr_steps = 0, 0
        model.train()  # prep model for training
        with tqdm(total=len(training_loader)) as t:
            for idx, batch in enumerate(training_loader):
                ids = batch['input_ids'].to(device, dtype=torch.long)
                mask = batch['attention_mask'].to(device, dtype=torch.long)
                labels = batch['labels'].to(device, dtype=torch.long)

                # loss, tr_logits = model(input_ids=ids, attention_mask=mask, labels=labels)
                outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
                loss = outputs[0]
                tr_logits = outputs[1]

                tr_loss += loss.item()

                nb_tr_steps += 1
                nb_tr_examples += labels.size(0)

                # compute training accuracy
                flattened_targets = labels.view(-1)  # shape (batch_size * seq_len,)
                active_logits = tr_logits.view(-1, model.num_labels)  # shape (batch_size * seq_len, num_labels)
                flattened_predictions = torch.argmax(active_logits, axis=1)  # shape (batch_size * seq_len,)

                # only compute accuracy at active labels
                active_accuracy = labels.view(-1) != -100  # shape (batch_size, seq_len)
                # active_labels = torch.where(active_accuracy, labels.view(-1), torch.tensor(-100).type_as(labels))

                labels = torch.masked_select(flattened_targets, active_accuracy)
                predictions = torch.masked_select(flattened_predictions, active_accuracy)

                tmp_tr_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
                tr_accuracy += tmp_tr_accuracy

                tmp_tr_f1 = f1_score(labels.cpu().numpy(), predictions.cpu().numpy(), average='micro')
                tr_f1 += tmp_tr_f1

                # gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    parameters=model.parameters(), max_norm=MAX_GRAD_NORM
                )

                # bar
                t.set_description(desc="Epoch %i" % epoch)
                t.set_postfix(steps=nb_tr_steps, accuracy=tr_accuracy / nb_tr_steps)
                t.update(1)

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        train_losses.append(tr_loss / nb_tr_steps)
        train_acces.append(tr_accuracy / nb_tr_steps)
        train_f1es.append(tr_f1 / nb_tr_steps)


        ######################
        # validate the model #
        ######################
        model.eval()  # prep model for evaluation

        valid_loss, valid_accuracy, valid_f1 = 0, 0, 0
        nb_valid_examples, nb_valid_steps = 0, 0

        with torch.no_grad():
            with tqdm(total=len(validation_loader)) as t:
                for idx, batch in enumerate(validation_loader):
                    ids = batch['input_ids'].to(device, dtype=torch.long)
                    mask = batch['attention_mask'].to(device, dtype=torch.long)
                    labels = batch['labels'].to(device, dtype=torch.long)

                    outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
                    loss = outputs[0]
                    valid_logits = outputs[1]

                    valid_loss += loss.item()

                    nb_valid_steps += 1
                    nb_valid_examples += labels.size(0)

                    # compute evaluation accuracy
                    flattened_targets = labels.view(-1)  # shape (batch_size * seq_len,)
                    active_logits = valid_logits.view(-1, model.num_labels)  # shape (batch_size * seq_len, num_labels)
                    flattened_predictions = torch.argmax(active_logits, axis=1)  # shape (batch_size * seq_len,)

                    # only compute accuracy at active labels
                    active_accuracy = labels.view(-1) != -100  # shape (batch_size, seq_len)

                    labels = torch.masked_select(flattened_targets, active_accuracy)
                    predictions = torch.masked_select(flattened_predictions, active_accuracy)

                    tmp_valid_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
                    valid_accuracy += tmp_valid_accuracy
                    # F1 score
                    tmp_valid_f1 = f1_score(labels.cpu().numpy(), predictions.cpu().numpy(), average='micro')
                    valid_f1 += tmp_valid_f1

                    t.set_description(desc="Validation on the validation set")
                    t.set_postfix(steps=idx, acc= valid_accuracy/nb_valid_steps)
                    t.update(1)



        valid_losses.append(valid_loss / nb_valid_steps)
        valid_acces.append(valid_accuracy / nb_valid_steps)
        valid_f1es.append(valid_f1 / nb_valid_steps)

        epoch_len = len(str(n_epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'train_loss: {tr_loss / nb_tr_steps:.5f} ' +
                     f'valid_loss: {valid_loss / nb_valid_steps:.5f}')

        print(print_msg)

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_accuracy / nb_valid_steps, model)

        if early_stopping.early_stop:
            print("Early stopping and refit!")
            refit(model, save_dir, all_loader, device)
            break

        if epoch >= n_epochs:
            print("Maximum epoch reached and refit!")
            refit(model, save_dir, all_loader, device)
            break

    final_outputs = (train_losses, train_acces, train_f1es, valid_losses, valid_acces, valid_f1es)

    return model, final_outputs

def plot_model(results, model_dir):
    results['epoch'] = range(1,results.shape[0]+1)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(results.epoch, results.train_loss, label = 'train')
    plt.plot(results.epoch, results.valid_loss, label='validation')
    plt.legend()
    plt.title('Loss in each epoch')
    plt.savefig(os.path.join(model_dir,'LossPlot.png'))

    plt.figure()
    plt.plot(results.epoch, results.train_accuracy, label = 'train')
    plt.plot(results.epoch, results.valid_accuracy, label='validation')
    plt.legend()
    plt.title('Accuracy in each epoch')
    plt.savefig(os.path.join(model_dir,'AccuracyPlot.png'))

    plt.figure()
    plt.plot(results.epoch, results.train_f1, label = 'train')
    plt.plot(results.epoch, results.valid_f1, label= 'validation')
    plt.legend()
    plt.title('Micro-averaged F1 in each epoch')
    plt.savefig(os.path.join(model_dir,'F1Plot.png'))


def refit(model, save_dir, all_loader, device):
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(save_dir+'/best_network.pth'))
    # retrain the model with all the training data and validation data
    model.train()
    with tqdm(total=len(all_loader)) as t:
        for idx, batch in enumerate(all_loader):
            ids = batch['input_ids'].to(device, dtype=torch.long)
            mask = batch['attention_mask'].to(device, dtype=torch.long)
            labels = batch['labels'].to(device, dtype=torch.long)
            outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
            t.set_description(desc="Refit on the combined datasets")
            t.set_postfix(steps=idx + 1)
            t.update(1)
    torch.save(model, save_dir+'/best_network_refit.pth')
    #torch.save(model.state_dict(),, save_dir + '/best_network_refit.pth')


if __name__ == '__main__':
    config_path= '../config/config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

        common = config['common']
        meta_prefix = common['meta_prefix']
        tagging_scheme = common['tagging_scheme']
        labels_path = common['labels_path']
        dict_dir = common['dict_dir']
        model_name = common['model_name']
        # rules = common['rules']
        sent_seg = common['sent_seg']
        maxlen = common['maxlen']
        transformer_name = common['transformer_name']
        segment = common['segment']

        train_cfg = config['train']
        out_dir = train_cfg['out_dir']
        n_epoch = train_cfg['n_epoch']
        batch_size = train_cfg['batch_size']
        hidden_layer_size = train_cfg['hidden_layer_size']
        patience = train_cfg['patience']
        train_valid_split = train_cfg['train_valid_split']
        ignore_O = train_cfg['ignore_O']

    sent_joined_train = os.path.join(out_dir,'tagged_sentences_joined'+'ignoer_O'*ignore_O+'_train.csv')
    sent_joined_valid = os.path.join(out_dir, 'tagged_sentences_joined' + 'ignoer_O' * ignore_O + '_valid.csv')

    trainBERT(sent_joined_train, sent_joined_valid, labels_path, out_dir, model_name, transformer_name, segment,
              maxlen, batch_size, n_epoch, patience)

