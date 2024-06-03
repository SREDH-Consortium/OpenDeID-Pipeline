import os
import datetime
import random
import string
from lxml import etree
import json
from dateutil.parser import parse
import argparse
from word2number import w2n

class Surrogator:
    def __init__(self, alpha_shift=10):
        self.alpha_shift = alpha_shift
        self.get_surr_dict('/input/dictionaries')
        self.reset()

    def reset(self):
        self.date_shift = random.randint(1, 730)
        self.pos_shift = 0
        self.surr_map = {}
        self.text = None
        
    def get_surr_dict(self, dict_dir):
        self.surr_dict = {}
        self.surr_dict.update({'MALE':{}})
        self.surr_dict.update({'FEMALE':{}})
        self.surr_dict.update({'LASTNAME':{}})
        male_names = open(os.path.join(dict_dir, 'male_names_unambig.txt')).read().strip().split('\n')
        female_names = open(os.path.join(dict_dir, 'female_names_unambig.txt')).read().strip().split('\n')
        for name in male_names:
            if self.surr_dict['MALE'].get(name[0].upper()) is not None:
                self.surr_dict['MALE'][name[0].upper()].append(name)
            else:
                self.surr_dict['MALE'][name[0].upper()] = [name]
        for name in female_names:
            if self.surr_dict['FEMALE'].get(name[0].upper()) is not None:
                self.surr_dict['FEMALE'][name[0].upper()].append(name)
            else:
                self.surr_dict['FEMALE'][name[0].upper()] = [name]
        last_names = open(os.path.join(dict_dir, 'last_names_unambig.txt')).read().strip().split('\n')
        for name in last_names:
            if self.surr_dict['LASTNAME'].get(name[0].upper()) is not None:
                self.surr_dict['LASTNAME'][name[0].upper()].append(name)
            else:
                self.surr_dict['LASTNAME'][name[0].upper()] = [name]
        
        self.surr_dict['STATE'] = open(os.path.join(dict_dir, 'us_states.txt')).read().strip().split('\n')
        self.surr_dict['COUNTRY'] = open(os.path.join(dict_dir, 'countries_unambig.txt')).read().strip().split('\n')
        self.surr_dict['CITY'] = open(os.path.join(dict_dir, 'australia_cities.txt')).read().strip().split('\n')
        self.surr_dict['HOSPITAL'] = open(os.path.join(dict_dir, 'stripped_hospitals.txt')).read().strip().split('\n')
        self.surr_dict['ORGANIZATION'] = open(os.path.join(dict_dir, 'company_names_unambig.txt')).read().strip().split('\n')
        self.surr_dict['OCCUPATIONS'] = open(os.path.join(dict_dir, 'occupations.txt')).read().strip().split('\n')
        self.surr_dict['STREET'] = open(os.path.join(dict_dir, 'streets.txt')).read().strip().split('\n')
        self.surr_dict['DEPARTMENT'] = open(os.path.join(dict_dir, 'departments_hsa.txt')).read().strip().split('\n')
        self.surr_dict['STATE_ABBR'] = {'NEW SOUTH WALES': 'NSW', 'QUEENSLAND': 'QLD', 'SOUTH AUSTRALIA': 'SA', 'VICTORIA': 'VIC', 'WESTERN AUSTRALIA': 'WA',
                                    'NORTHERN TERRITORY': 'NT', 'AUSTRALIAN CAPITAL TERRITORY': 'ACT'}

    def gen_surrogate(self, tag_text, tag):
        if tag in ('PATIENT', 'DOCTOR'):
            if tag_text.upper().startswith(('DR.', 'MR.', 'MISS.', 'MRS')):
                lookup, surr = self.get_name_surrogate(tag_text.split('.', 1)[1].strip())
                surr = tag_text.split('.', 1)[0] + '.' + surr
            else:
                lookup, surr = self.get_name_surrogate(tag_text)
            if len(surr.replace(" ", "")) > 2:
                if ',' in surr:
                    self.surr_map[lookup] = ' '.join(surr.split(',')[::-1])
                else:
                    self.surr_map[lookup] = surr
            return surr
        elif tag_text in self.surr_map:
            surr = self.surr_map[tag_text]
        elif tag == 'STATE':
            if tag_text.upper() in self.surr_dict['STATE_ABBR']:
                abbr = self.surr_dict['STATE_ABBR'][tag_text.upper()]
                if abbr in self.surr_map:
                    surr = self.surr_map[abbr]
                else:
                    surr = random.choice(list(self.surr_dict['STATE_ABBR'].keys()))
                    self.surr_map[self.surr_dict['STATE_ABBR'][tag_text.upper()]] = surr
            else:
                surr = random.choice(list(self.surr_dict['STATE_ABBR'].keys()))
                self.surr_map[tag_text.upper()] = surr

        elif tag in ('COUNTRY', 'HOSPITAL', 'CITY', 'ORGANIZATION', 'ORGANIZATION'):
            surr = random.choice(self.surr_dict.get(tag))
        elif tag == "DATE":
            surr = self.get_date_surrogate(tag_text)
        elif tag == "ROOM":
            surr = str(random.randrange(100,2000))  # to be modified
        elif tag == "ZIP":
            surr = str(random.randrange(3000,9999))
        elif tag == "BIOID":  # to be modified
            surr =  str(random.randrange(11,19)) + "N" + str(random.randrange(1000000, 9999999)) # todo
        elif tag == "IDNUM":
            surr = str(random.randrange(11,19)) + "R" + str(random.randrange(1000000, 9999999))
        elif tag == "AGE":
            if not tag_text.isdigit():
                try:
                    w2v.word2num(tag_text)
                except:
                    print("Invalid age string %s" % tag_text)
                    surr = "[** AGE **]"
                    return surr
            surr = str(int(tag_text) + self.date_shift // 365)
        elif tag == "MEDICALRECORD":
            surr = str(random.randrange(1000000, 9999999)) + '.' + random.choice(['RAN', 'PWP', 'STG'])
        elif tag == "USERNAME":
            surr = random.choice(string.ascii_letters) + random.choice(string.ascii_letters) + str(random.randrange(1000, 9999)) # todo
        elif tag == "EMAIL":
            surr = "%s@gmail.com"%uuid.uuid1().hex
        elif tag == "PROFESSION":
            surr = self.surr_map.get(tag_text)
            if surr is None:
                surr = random.choice(self.surr_dict['OCCUPATIONS'])
        elif tag in ("FAX", "PHONE"):
            surr = str(random.choice(1,8)).zfill(2) + str(random.choice(0, 9999999)).zfill(7)
        elif tag == "DEVICE":
            surr = "[**%s**]"%tag # todo
        elif tag == "HEALTHPLAN":
            surr = "[**%s**]"%tag # todo
        elif tag == "URL":
            surr = "www."
            for _ in random.randrange(3,8):
                surr += random.choice(string.ascii_letters)
            surr += ".com"
        elif tag == "LICENSE":
            surr = "[**%s**]"%tag # todo
        elif tag == "STREET":
            surr = random.choice(self.surr_dict['STREET'])
        else:
            surr = "[**%s**]"%tag  # todo
        self.surr_map[tag_text] = surr
        return surr

    def get_name_surrogate(self, name):
        alpha_shift = self.alpha_shift % 26
        reverse = False

        if ',' in name:  # LAST, FIRST
            reverse = True
            name = ' '.join([i.strip() for i in name.split(',')[::-1]])

        if not name.replace(" ", "").isalpha():
            return name, random.choice(self.surr_dict['LASTNAME'][random.choice(string.ascii_letters).upper()])

        lookup = ''.join([i + ' ' if len(i) == 1 else i[:2] for i in name.split(' ')]).upper()
        existing = self.surr_map_match(lookup)
        if existing is not None:
            surr = existing
            for part in range(0, len(lookup), 2):
                if len(lookup[part: part+2].replace(" ", "")) < 2:
                    existing = existing.split(" ")
                    existing[part//2] = self.alpha_shifting(lookup[part: part+2].replace(" ", ""), self.alpha_shift)
                    surr = " ".join(existing)
                    continue
            if reverse:
                surr = ', '.join(surr.split(" ")[::-1])
            return lookup, surr

        if ' ' in name: # FIRST LAST
            part = name.split(' ')
            for i in range(len(part)):
                if len(part[i]) == 1 and part[i].isupper():
                    part[i] = self.alpha_shifting(part[i], self.alpha_shift)
                else:
                    if i == len(part) - 1:
                        initial = part[i][0].upper()
                        shift_initial = self.alpha_shifting(initial, self.alpha_shift)
                        # while surr_dict['LASTNAME'].get(shift_initial) is None:
                        #     shift_initial = chr(ord(shift_initial)+1)
                        part[i] = random.choice(self.surr_dict['LASTNAME'][shift_initial])
                    else:
                        sex = None
                        for gender in ['MALE', 'FEMALE']:
                            initial = part[i][0].upper()
                            if part[i].upper() in self.surr_dict[gender][initial]:
                                sex = gender
                                break
                        if sex is None:
                            sex = random.choice(['MALE', 'FEMALE'])
                        shift_initial = self.alpha_shifting(initial, alpha_shift)
                        part[i] = random.choice(self.surr_dict[sex][shift_initial])
            lookup = ''.join([p[:2].upper() for p in part])
            if reverse:
                surr = ','.join(part[::-1])
            else:
                surr = ' '.join(part)
        
        else:  # ONLY FIRST OR LAST
            sex = None
            for gender in ['MALE', 'FEMALE']:
                initial = name[0].upper()
                if name.upper() in self.surr_dict[gender][initial]:
                    sex = gender
                    break
            if sex is None:
                sex = random.choice(['MALE', 'FEMALE'])
            shift_initial = self.alpha_shifting(initial, alpha_shift)
            surr = random.choice(self.surr_dict[sex][shift_initial])
            lookup = surr[:2].upper() + '  '

        if name.isupper():  # A B or A LAST or FIRST B
            if len(name.replace(' ', '')) == 1:
                surr = self.alpha_shifting(name, alpha_shift)
            if len(name.replace(' ', '')) == 2:
                surr = ''.join([self.alpha_shifting(i, alpha_shift) if i.isalpha() else i for i in name])
        return lookup, surr


    def surr_map_match(self, lookup):
        for name in self.surr_map:
            if len(lookup) == 2:
                if lookup == name[:2]:
                    return self.surr_map[name].split(' ')[0]
                elif lookup == name[2:]:
                    return self.surr_map[name].split(' ')[1]
            else:
                if lookup == name or (lookup[:2] == name[:2] and lookup[2] == name[2]) or (lookup[0] == name[0] and lookup[2:] == name[2:]):
                    return self.surr_map[name]
        return None

    def alpha_shifting(self, alpha, alpha_shift):
        alpha_shift %= 26
        shift = ord(alpha) + alpha_shift
        return chr(ord('A') * (shift // 90) + shift % 90)

    def get_date_surrogate(self, date_str):
        try:
            date = parse(date_str, dayfirst=True)
            surr = date + datetime.timedelta(self.date_shift)
            return datetime.datetime.strftime(surr, "%Y-%m-%d")
        except:
            print("cannot parse date string %s" % date_str)
            return "[** DATE **]"

    def surrogation_replacing(self, surrogation, start, end):
        self.text = self.text[:start+self.pos_shift] + str(surrogation) + self.text[end+self.pos_shift:]
        self.pos_shift += len(surrogation) - (end - start)

    def file_surrogate(self, file_path, out_path):
        self.reset()
        tree = etree.parse(file_path)
        root = tree.getroot()
        self.text = root[0].text
        tags = root[1]
        
        for tag in tags:
            # <ID id="P0" start="14" end="24" text="13R379532O" TYPE="IDNUM" comment=""/>
            tag_type = tag.get('TYPE')
            tag_text = tag.get('text')
            start = int(tag.get('start'))
            end = int(tag.get('end'))
            surrogation = self.gen_surrogate(tag_text, tag_type)
            self.surrogation_replacing(surrogation, start, end)

        if not os.path.exists(out_path):
            try:
                os.makedirs(out_path)
            except OSError as e:
                # check for race condition (i.e. if dirs were made in between)
                if e.errno != errno.EEXIST:
                    raise # other error
        with open(os.path.join(out_path, os.path.basename(file_path.replace('xml', 'txt'))), 'w') as surrogated:
            surrogated.write(self.text)

    def multi_file_surrogate(self, file_dir, out_path):
        for file in os.listdir(file_dir):
            if file.endswith('.xml'):
                file_path = os.path.join(file_dir, file)
                self.file_surrogate(file_path, out_path)


if __name__ == '__main__':
    import os
    if os.getcwd().split('\\')[-1] != 'PyDLNER':
        if 'PyDLNER' in os.listdir():
            os.chdir('PyDLNER')
    print(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', action='store_const', const=1, default=0)
    parser.add_argument('--in', dest='in_path', required=True)
    parser.add_argument('--out', dest='out_path', required=True)
    args = parser.parse_args()

    in_path = args.in_path
    out_path = args.out_path
    surrogator = Surrogator()
    if args.r == 1:
        surrogator.multi_file_surrogate(in_path, out_path)
    else:
        surrogator.file_surrogate(in_path, out_path)

