import random

def preprocess_orgtrain():
    # 1. Preprocess CSV file
    list_label = []
    list_text = []
    cnt = 0
    with open('train_yelp.csv', 'r') as f:
        for line in f:
            if 'http' in line:
                continue
            if line[0] not in ['0', '1']:
                continue
            label = line[0]             # label part
            line = line[2:].strip()     # string part

            line = line.replace(r"\n", " ")
            line = line.replace("!", "")
            line = line.replace(r"\"", "")
            line = line.replace('"', "")
            line = line.replace('...', " ")
            line = line.replace('.', " .")
            line = line.replace(',', " ,")
            if not line:
                continue
            if line[0] == '"' and line[-1] == '"':
                line = line[1:-1]
            if line[-1] == '.':
                line = line[:-1].strip()

            # discard too-long texts
            if len(line.split()) > 30 or len(line.split()) < 5:
                continue

            # print(line)
            # cnt += 1
            # if cnt == 15:
            #     break

            list_label.append(label)
            list_text.append(line.lower())

    # 2. Training and Validation split
    list_label_train, list_label_val = [], []
    list_text_train, list_text_val = [], []
    num_total = len(list_label)
    val_ids = random.sample(list(range(num_total)), int(num_total*0.1))
    for i in range(num_total):
        if i in val_ids:
            list_label_val.append(list_label[i])
            list_text_val.append(list_text[i])
        else:
            list_label_train.append(list_label[i])
            list_text_train.append(list_text[i])

    # 3. Write into train.tsv and dev.tsv
    with open("train.tsv", "w") as f:
        for text, label in zip(list_text_train, list_label_train):
            f.write("%s\t%s\n" % (text, label))
    with open("dev.tsv", "w") as f:
        for text, label in zip(list_text_val, list_label_val):
            f.write("%s\t%s\n" % (text, label))


def preprocess_orgtest():
    # 1. Preprocess CSV file
    list_label = []
    list_text = []
    cnt = 0
    with open('test_yelp.csv', 'r') as f:
        for line in f:
            if 'http' in line:
                continue
            if line[0] not in ['0', '1']:
                continue
            label = line[0]  # label part
            line = line[2:].strip()  # string part

            line = line.replace(r"\n", " ")
            line = line.replace("!", "")
            line = line.replace(r"\"", "")
            line = line.replace('"', "")
            line = line.replace('...', " ")
            line = line.replace('.', " .")
            line = line.replace(',', " ,")
            if not line:
                continue
            if line[0] == '"' and line[-1] == '"':
                line = line[1:-1]
            if line[-1] == '.':
                line = line[:-1].strip()

            # discard too-long texts
            if len(line.split()) > 30 or len(line.split()) < 5:
                continue

            # print(line)
            # cnt += 1
            # if cnt == 15:
            #     break

            list_label.append(label)
            list_text.append(line.lower())

    # 2. Write into test.tsv
    with open("test.tsv", "w") as f:
        for text, label in zip(list_text, list_label):
            f.write("%s\t%s\n" % (text, label))


if __name__ == "__main__":
    preprocess_orgtrain()

    preprocess_orgtest()