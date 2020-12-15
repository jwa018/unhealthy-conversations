from transformers import BertForSequenceClassification, BertTokenizer
import torch as th
import pandas as pd
from main import get_clean_df, MultiLabelDataset
from torch.utils.data import DataLoader
from torch import cuda
import numpy as np
from sklearn.metrics import roc_curve, auc
import os
import json
# import matplotlib.pyplot as plt

def validation(testing_loader, model, device):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with th.no_grad():
        for data in testing_loader:
            ids = data['ids'].to(device, dtype = th.long)
            mask = data['mask'].to(device, dtype = th.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = th.long)
            targets = data['targets'].to(device, dtype = th.float)
            outputs = model(ids, mask, token_type_ids)[0]
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(th.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets


if __name__ == '__main__':
    csv = pd.read_csv('corpus/val.csv')
    data = get_clean_df(csv)
    tokenizer = BertTokenizer.from_pretrained('models/new_save', do_lower_case=True)
    model = BertForSequenceClassification.from_pretrained('models/new_save')

    device = 'cuda' if cuda.is_available() else 'cpu'
    model.to(device)
    MAX_LEN = 128
    BATCH_SIZE = 64
    classes = ['antagonise', 'condescending', 'dismissive', 'generalisation_unfair', 'hostile', 'sarcastic', 'healthy']
    dataset = MultiLabelDataset(data, tokenizer, MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    output, targets = validation(dataloader, model, device)
    np_scores = np.array(output)
    np_targets = np.array(targets)
    try:
        os.mkdir('out')
    except:
        pass
    out_dict = dict()
    out_dict['scores'] = output
    out_dict['targets'] = targets
    with open('out/val_res.json', 'w') as f:
        json.dump(out_dict, f)
    exit()
    roc_auc_dict = dict()

    for i in range(np_targets.shape[1]):
        tmp_dict = dict()
        pred = np_scores[:, i]
        target = np_targets[:, i]
        fpr, tpr, thresholds = roc_curve(target, pred, pos_label=1)
        auc_score = auc(fpr, tpr)
        tmp_dict['auc'] = auc_score
        tmp_dict['fpr'] = fpr
        tmp_dict['tpr'] = tpr
        tmp_dict['thresholds'] = thresholds
        roc_auc_dict[classes[i]] = tmp_dict
    # plt.figure()
    for item in roc_auc_dict.items():
        print(item)
        print(type(item))
        print(item['auc'])