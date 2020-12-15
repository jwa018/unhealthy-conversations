import matplotlib.pyplot as plt
import numpy as np
import json
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib
matplotlib.use('TkAgg')


if __name__ == '__main__':
    with open('out/val_res.json', 'r') as f:
        res_dict = json.load(f)
    scores = np.array(res_dict['scores'])
    targets = np.array(res_dict['targets'])
    classes = ['antagonise', 'condescending', 'dismissive', 'generalisation_unfair', 'hostile', 'sarcastic', 'healthy']
    roc_dict = dict()
    for i in range(len(scores[0])):
        tmp_dict = dict()
        target = targets[:,i]
        score = scores[:,i]
        fpr, tpr, _ = roc_curve(target, score)
        auc_score = auc(fpr, tpr)
        tmp_dict['fpr'] = fpr
        tmp_dict['tpr'] = tpr
        tmp_dict['auc'] = auc_score
        roc_dict[classes[i]] = tmp_dict
    plt.figure()
    for cls_name in classes:
        tmp_dict = roc_dict[cls_name]
        fpr = tmp_dict['fpr']
        tpr = tmp_dict['tpr']
        auc_score = tmp_dict['auc']
        plt.plot(fpr, tpr, label=f'{cls_name} {auc_score:.2f}')
    plt.legend(loc='lower right')
    plt.savefig('figures/val.png')
    plt.show()