import numpy as np
import pandas as pd
from sklearn import metrics
import transformers
import torch as th
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from torch import cuda
import wandb
import os
from dataset import UCCDataset

def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        acc_list.append(tmp_a)
    return np.mean(acc_list)


def loss_fn(outputs, targets):
    return th.nn.BCEWithLogitsLoss()(outputs, targets)


def train(n_epoch, log_interval):
    model.train()
    log_counter = 0
    for epoch in range(n_epoch):
        loss_list = list()
        for i, data in enumerate(train_loader):
            ids = data['ids'].to(device, dtype=th.long)
            mask = data['mask'].to(device, dtype=th.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=th.long)
            targets = data['targets'].to(device, dtype=th.float)

            outputs = model(ids, mask, token_type_ids)[0]

            loss = loss_fn(outputs, targets)
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if log_counter % log_interval == 0:
                mean_epoch_loss = np.mean(loss_list)
                val_output, val_target = validation(val_loader)
                val_output = np.array(val_output) >= 0.5
                val_ham_loss = metrics.hamming_loss(val_target, val_output)
                val_ham_score = hamming_score(np.array(val_target), np.array(val_output))
                wandb.log({
                    'Epoch': epoch,
                    'Mean Train Loss': mean_epoch_loss,
                    'Hamming loss': val_ham_loss,
                    'Hamming score': val_ham_score
                })
            log_counter += 1


def validation(testing_loader):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
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


if __name__ == "__main__":
    device = 'cuda' if cuda.is_available() else 'cpu'
    datasets = list()
    for subset in ['train', 'val', 'test']:
        data = pd.read_csv(f'data/{subset}.csv')
        datasets.append(get_clean_df(data))
    
    train_data, val_data, test_data = datasets

    MAX_LEN = 128
    TRAIN_BATCH_SIZE = 64
    VALID_BATCH_SIZE = 64
    EPOCHS = 1
    LEARNING_RATE = 1e-05
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                              truncation=True,
                                              do_lower_case=True)
    tokenizer.save_pretrained('models/new_save')
    exit()

    print("TRAIN Dataset: {}".format(train_data.shape)) 
    print("VAL Dataset: {}".format(val_data.shape))
    print("TEST Dataset: {}".format(test_data.shape))

    training_set = UCCDataset(train_data, tokenizer, MAX_LEN)
    validation_set = UCCDataset(val_data, tokenizer, MAX_LEN)
    testing_set = UCCDataset(test_data, tokenizer, MAX_LEN)
    
    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    test_params = {'batch_size': VALID_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    train_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(testing_set, **test_params)
    config = BertConfig(
        name_or_path='bert-base-uncased',
        num_labels=7,
        hidden_dropout_prob=0.5,
        attention_probs_dropout_prob=0.5
        )
    model = BertForSequenceClassification(config)
    model.to(device)

    optimizer = th.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    wandb.init(project='Unhealthy Conversations')
    train(3, log_interval = 100)  
    # val_output, val_targets = validation(val_loader)
    # val_hamming_loss = metrics.hamming_loss(targets, final_outputs)
    # val_hamming_score = hamming_score(np.array(targets), np.array(final_outputs))
    # wandb.log({
    #     'Val Hamming Loss': val_hamming_loss,
    #     'Val Hamming Score': val_hamming_score
    # })
    try:
        os.mkdir('models')
    except:
        pass
    # output_model_file = './models/trained_model.bin'
    # output_vocab_file = './models/vocab.bin'

    # th.save(model, output_model_file)
    # tokenizer.save_vocabulary(output_vocab_file)
    try:
        os.mkdir('models/new_save')
    except:
        pass
    model.save_pretrained('models/new_save')