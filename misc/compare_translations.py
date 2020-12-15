with open('no_corpus/norwegian_train.txt', 'r') as f:
    lines = f.readlines()
with open('no_corpus/norwegian_train_2.txt', 'r') as f:
    lines2 = f.readlines()

disimmilar = 0
for line1, line2 in zip(lines, lines2):
    if line1 != line2:
        disimmilar += 1
print(disimmilar)