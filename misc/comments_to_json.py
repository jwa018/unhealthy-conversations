import csv
import json
from tqdm import tqdm
import atexit 

# translator = Translator()
# a = translator.translate('jeg heter Joakim')
# print(a.text)


def save_file(dataset, file_to_save):
    dct = {'thing': file_to_save}
    with open(f'no_corpus/{dataset}.json', 'w') as f:
        json.dump(dct, f)


if __name__ == '__main__':
    dataset = 'test'
    comments = list()
    n_char = 0
    with open(f'corpus/{dataset}.csv', 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        idx = 0
        for line in csv_reader:
            if idx != 0:
                comment = line[2]
                n_char += len(comment)
                comments.append(comment)
            idx += 1
    print(n_char)
    json_tmp = {'comments': comments}
    with open(f'no_corpus/{dataset}.json', 'w') as f:
        json.dump(json_tmp, f)