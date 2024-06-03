import logging
from collections import defaultdict


def freq(in_path, out_path, meta_prefix, normalize=False):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)s %(levelname)s %(asctime)s: %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info('Started logger')

    logger.info('Counting')
    freqs = defaultdict(int)
    input = open(in_path, 'r')
    for line in input:
        token, _, offset = line.strip().partition('\t')
        if line.startswith(meta_prefix):
            continue
        # should I normalize e.g. lowercase?
        freqs[token] += 1
    input.close()

    logger.info('Writing')
    output = open(out_path, 'w')
    for token, count in sorted(freqs.items(), key=lambda x: x[1], reverse=True):
        output.write('%s\t%s\n' % (token, count))
    output.close()
    return out_path

if __name__ == '__main__':
    # get args from command line or config
    freq()
