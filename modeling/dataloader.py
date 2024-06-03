from torch.utils.data import Dataset
import numpy as np
import torch

class dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, labels_to_ids):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels_to_ids = labels_to_ids

    def __getitem__(self, index):
        # step 1: get the sentence and word labels
        sentence = self.data.text[index].strip().split()
        word_labels = self.data.tags[index].split(",")

        # step 2: use tokenizer to encode sentence (includes padding/truncation up to max length)
        # BertTokenizerFast provides a handy "return_offsets_mapping" functionality for individual tokens
        encoding = self.tokenizer(sentence,is_split_into_words=True,
                                  return_offsets_mapping=True,
                                  padding='max_length',
                                  truncation=True,
                                  max_length=self.max_len)

        # step 3: create token labels only for first word pieces of each tokenized word
        labels = [self.labels_to_ids[label] for label in word_labels]
        # code based on https://huggingface.co/transformers/custom_datasets.html#tok-ner
        # create an empty array of -100 of length max_length
        encoded_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100

        # set only labels whose first offset position is 0 and the second is not 0
        i = 0
        for idx, mapping in enumerate(encoding["offset_mapping"]):
            if mapping[0] == 0 and mapping[1] != 0:
                # overwrite label
                encoded_labels[idx] = labels[i]
                i += 1

        # step 4: turn everything into PyTorch tensors
        item = {key: torch.LongTensor(val) for key, val in encoding.items()}
        # item['labels'] = torch.as_tensor(encoded_labels)
        item['labels'] = torch.LongTensor(encoded_labels)

        return item

    def __len__(self):
        return self.len

