import torch as th
from torch.utils.data import Dataset
import pandas as pd
from transformers import BertTokenizer


def get_clean_df(raw_df):
    new_df = pd.DataFrame()
    new_df['text'] = raw_df['comment']
    new_df['labels'] = raw_df[
        ['antagonise',
         'condescending',
         'dismissive',
         'generalisation_unfair',
         'hostile',
         'sarcastic',
         'healthy']
        ].values.tolist()
    return new_df


class UCCDataset(Dataset):

    def __init__(self, csv_file, tokenizer, max_len):
        self.data = get_clean_df(pd.read_csv(csv_file))
        self.text = self.data.text
        self.targets = self.data.labels
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = str(self.text[idx])
        text = " ".join(text.split())
        inputs = self.tokenizer(
            text=text,
            text_pair=None,
            add_special_tokens=True,
            padding='max_length',
            max_length=self.max_len,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']

        return {
            'ids': th.tensor(ids, dtype=th.long),
            'mask': th.tensor(mask, dtype=th.long),
            'token_type_ids': th.tensor(token_type_ids, dtype=th.long),
            'targets': th.tensor(self.targets[idx], dtype=th.float)
        }
