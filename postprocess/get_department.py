#-This file will generate the departments_hsa.txt dictionary for the surrogate generation. 
from lxml import etree
import os
import re

departments = []
dirs = ('test-gold', 'train-valid')

for d in dirs:
    for file in os.listdir('../input/datasets/HSA/Deid/' + d):
        if file.endswith('.xml'):
            file_path = os.path.join('../input/datasets/HSA/Deid/', d, file)
            tree = etree.parse(file_path)
            root = tree.getroot()
            tags = root[1]
            for tag in tags:
                if tag.get('TYPE').upper() == 'DEPARTMENT':
                    department = tag.get('text')
                    if department not in departments:
                        pat = re.compile(r'^\d\.\d?')
                        department = re.sub(pat, '', department)
                        departments.append(department)

with open('departments_hsa.txt', 'w') as f:
    f.write('\n'.join(departments))