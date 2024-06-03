#-*-coding:utf-8-*-
import os

from lxml import etree

categories = {'NAME':{}, 'PROFESSION':0, 'LOCATION':{}, 'AGE': 0, 'DATE': 0,
                'CONTACT': {}, 'IDs':{}, 'Other': 0}
NAMES = ['PATIENT', 'DOCTOR', 'USERNAME']
LOCATIONS = ['ROOM', 'DEPARTMENT', 'HOSPITAL', 'ORGANIZATION', 'STREET', 'CITY', 'STATE', 
'COUNTRY', 'ZIP', 'LOCATION-OTHER']
CONTACT = ['PHONE', 'FAX', 'EMAIL', 'URL', 'IPADDRESS']
IDs = ['SSN', 'MEDICALRECORD', 'HEALTHPLAN', 'ACCOUNT',
      'LICENSE', 'VEHICLE', 'DEVICE', 'BIOID', 'IDNUM']
sub_categories = {}
sub_categories.update({k: 'NAME' for k in NAMES})
sub_categories.update({k: 'LOCATION' for k in LOCATIONS})
sub_categories.update({k: 'CONTACT' for k in CONTACT})
sub_categories.update({k: 'IDs' for k in IDs})
for k,v in sub_categories.items():
    categories[v][k] = 0

# specify path while needed

for filename in [f for f in os.listdir() if f.endswith('.xml')]:
    xml = etree.parse(filename)
    root = xml.getroot()
    tags = root.find('TAGS')

    for tag in tags:
        cate = tag.get('TYPE')
        if cate is not None:
            if cate in sub_categories:
                categories[sub_categories[cate]][cate] += 1
            else:
                categories[cate] += 1

with open('stats.txt', 'w') as f:
    for cate, count in categories.items():
        if isinstance(count, dict):
            for i,j in count.items():
                f.write(':'.join((cate,i,str(j))) + '\n')
        else:
            f.write(':'.join((cate, str(count))) + '\n')

