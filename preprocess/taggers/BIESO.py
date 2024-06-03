import logging


def tag(in_path, out_path, meta_prefix):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)s %(levelname)s %(asctime)s: %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info('Started logger')

    logger.info('Open files')
    with open(in_path, 'r') as f:
        input = f.readlines()
    output = open(out_path, 'w')

    logger.info('Tagging')
    tag = None
    for i, line in enumerate(input):
        line = line.strip()
        if line.startswith(meta_prefix):
            if line.startswith(meta_prefix + 'STARTTAG'):
                tag = line.split('\t')[1]
            elif line.startswith(meta_prefix + 'ENDTAG'):
                tag = None
            elif line.startswith(meta_prefix + 'STARTDOC') or line.startswith(meta_prefix + 'ENDDOC'):
                tag = None
                output.write(line + '\n')
        elif tag:
            if input[i-1].startswith(meta_prefix + 'STARTTAG') and input[i+1].startswith(meta_prefix + 'ENDTAG'):
                prefix = 'S'
            elif input[i-1].startswith(meta_prefix + 'STARTTAG'):
                prefix = 'B'
            elif input[i+1].startswith(meta_prefix + 'ENDTAG'):
                prefix = 'E'
            else:
                prefix = 'I'
            output.write('%s\t%s_%s\n' % (line.split('\t')[0], prefix, tag))
        else:
            token, _, offset = line.strip().partition('\t')
            if token =='':
                token="\""			    
            output.write('%s\tO\n' % token)
    output.close()
    return out_path

if __name__ == '__main__':
    # read options from command line?
    tag()