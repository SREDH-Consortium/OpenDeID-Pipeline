import pandas as pd
from math import ceil

# split sentence based on maxlen only.
def segment_sequence(origdata, maxlen = 128):
    # use sentence-based sequence
    # ID = sentID
    data = origdata[['sentID', 'text', 'tags']]
    data.rename(columns={'sentID': 'ID'}, inplace=True)

    maxlength = data['text'].apply(lambda x: len(x.split())).max()
    # this give how many substrings are required, e.g. if maxlen is 128, but we have sentences with 300 tokens maximum, this required 3 segmented sequences
    n = ceil(maxlength/maxlen)
    subtext_names = []
    subtags_names = []
    for i in range(n):
        subtext_names.append('subtext_'+str(i))
        subtags_names.append('subtags_' + str(i))
        data['subtext_'+str(i)] = data['text'].transform(lambda x: ' '.join(x.split()[(i*maxlen):((i+1)*maxlen)]))
        data['subtags_'+str(i)] = data['tags'].transform(lambda x: ','.join(x.split(',')[(i*maxlen):((i+1)*maxlen)]))

    datatext = data[['ID'] + subtext_names].set_index(['ID']).stack().rename('text').reset_index()
    datatags = data[['ID'] + subtags_names].set_index(['ID']).stack().rename('tags').reset_index()
    assert datatext.shape[0] == datatags.shape[0]
    datatext['tags'] = datatags['tags']
    datatext = datatext.drop(datatext[datatext.text==''].index).reset_index(drop=True)
    datatext['subID'] = datatext['level_1'].transform(lambda x: x.split('_')[1])
    return datatext[['ID','subID','text','tags']]



