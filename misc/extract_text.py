import csv

english_comments = list()
with open('corpus/train.csv', 'r', encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    idx = 0
    for line in csv_reader:
        idx += 1
        comment = line[2]
        english_comments.append(comment)

n_divisions = 2

for i in range(n_divisions):
    with open(f'{i}_train_comments.txt')

with open('test_comments.txt', 'w', encoding='utf-8') as f:
    for i, comment in enumerate(english_comments):
        f.write(comment + '\n')