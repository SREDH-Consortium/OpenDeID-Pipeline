import logging
from collections import deque, namedtuple


def format_ngram(ngram):
    """Formats an ngram into a string ready for writing.
    
    Args:
        ngram: Deque of `Tags`, which are named tuples containing a token and its label.

    Returns:
        A string representation of the input ngram.
    """
    tokens = '\t'.join([tag.token for tag in ngram])
    tag = ngram[len(ngram)//2].label
    return '%s\t%s\n' % (tokens, tag)

def write_ngrams(in_path, out_path, vocab_path, window_size, meta_prefix, threshold=0):
    """Process tagged tokens into ngrams.
    
    Args:
        in_path: Path to a input text file containing tagged tokens. Input file must be in the format of one pair
            of a token and a tag per line, tab-separated, excepting metadata lines.
        out_path: Path to output a text file containing tagged ngrams.
        vocab_path: Path to a vocabulary containing tokens and their frequencies.
        window_size: Size of the context window i.e. ngram will be of size 2 * window_size + 1.
        meta_prefix: Prefix of lines used for metadata in the input file, which will be ignored.
        threshold: Frequency cutoff; tokens with frequencies below the threshold will be substituted with 'UNK'.

    Returns:
        Path to the output text containing tagged ngrams.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)s %(levelname)s %(asctime)s: %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info('Started logger')

    logger.info('Reading %s' % in_path)
    with open(in_path, 'r') as f:
        input = f.readlines()
    logger.info('Reading vocab')
    with open(vocab_path, 'r') as f:
        vocab = {}
        for line in f:
            word, _, freq = line.strip().partition('\t')
            vocab[word] = int(freq)
    output = open(out_path, 'w')
    ngram_size = window_size * 2 + 1
    Tag = namedtuple('Tag', ['token', 'label'])
    pad_tag = Tag('PAD','O')
    ngram = deque([pad_tag, pad_tag], ngram_size)
    logger.info('Creating ngrams')
    for line in input:
        line = line.strip()
        if line.startswith(meta_prefix + 'ENDDOC'):
            # end of report so pad then reset
            for i in range(window_size):
                ngram.append(pad_tag)
                output.write(format_ngram(ngram))
            ngram = deque([pad_tag, pad_tag], ngram_size)
            continue
        elif not line.startswith(meta_prefix):
            token, _, label = line.partition('\t')
            if vocab[token] < threshold and label == 'O':
                token = 'UNK'
            ngram.append(Tag(token, label))

        if len(ngram) == ngram_size:
            output.write(format_ngram(ngram))
    output.close()
    return out_path

if __name__ == '__main__':
    in_path = 'input/old/train_bieso_new.txt'
    out_path = 'output/old/ngrams.txt'
    vocab_path = 'output/old/freq.txt'
    window_size = 2
    meta_prefix = '#$'
    write_ngrams(in_path, out_path, vocab_path, window_size, meta_prefix)