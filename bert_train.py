import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn import metrics
import transformers
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

def get_sorted_icd_list(icd_code_test):

    icd_code = icd_code_test.split('\n')
    icd_code.sort()
    return icd_code

def gen_bert_dataset(input_dataset_path, output_dataset_path):

    # load instruction data
    f = open(input_dataset_path)
    data = json.load(f)

    X = []
    y = []

    for case in data:

        # extract real and pred data
        icd_code_text = None
        for conversation in case['conversations']:
            if conversation['from'] == 'human':
                X.append(conversation['value'])
            if conversation['from'] == 'gpt':
                icd_code_text = conversation['value']

        icd_code = get_sorted_icd_list(icd_code_text)
        y.append(icd_code)

    # One-hot encode the lists
    multibinarizer = MultiLabelBinarizer()

    y_true_onehot = multibinarizer.fit_transform(y)

    f = open(output_dataset_path, "w")

    #["val_0","val_1","val_2","val_3","val_4","val_5","val_6","val_7"]
    val_list = []
    header = '"id","comment_text",'
    for x in range(len(y_true_onehot[0])):
        header = header + '"val_' + str(x) + '",'
        val_list.append('val_' + str(x))
    header = header[:-1] + '\n'

    f.write(header)

    # print(header)
    for idx, x in enumerate(X):
        record = '"' + str(idx) + '","' + X[idx] + '",'
        for flag in y_true_onehot[idx]:
            record = record + str(flag) + ','
        record = record[:-1] + '\n'
        # print(record)
        f.write(record)
    f.close()

    return len(y_true_onehot[0]), val_list, multibinarizer


parser = argparse.ArgumentParser(description='BERT NLP')
parser.add_argument('--train_dataset_path', type=str, default='train_dataset.json', help='location of dataset')
parser.add_argument('--test_dataset_path', type=str, default='test_dataset.json', help='location of dataset')
parser.add_argument('--base_model', type=str, default='bert-base-uncased', help='location of dataset')
parser.add_argument('--num_epoch', type=int, default=1, help='location of dataset')
parser.add_argument('--cuda_device', type=int, default=0, help='location of dataset')
args = parser.parse_args()

#prepare the data
gen_bert_dataset(args.train_dataset_path, 'path_train.csv')
label_count, val_list, multibinarizer = gen_bert_dataset(args.test_dataset_path, 'path_test.csv')

# save file
base_file = os.path.basename(args.train_dataset_path)
base_filename, base_file_extension = os.path.splitext(base_file)
model_name = 'custom-' + args.base_model + '_' + base_filename + '_epochs' + str(args.num_epoch)

# Defining some key variables that will be used later on in the training
MAX_LEN = 200
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = args.num_epoch
LEARNING_RATE = 1e-05
tokenizer = BertTokenizer.from_pretrained(args.base_model)

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.comment_text = dataframe.comment_text
        self.targets = self.data.list
        self.max_len = max_len

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
        comment_text = str(self.comment_text[index])
        comment_text = " ".join(comment_text.split())

        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

# Creating the dataset and dataloader for the neural network


train_size = 0.8

df = pd.read_csv("path_train.csv")
df['list'] = df[df.columns[2:]].values.tolist()
train_dataset = df[['comment_text', 'list']].copy()

df = pd.read_csv("path_test.csv")
df['list'] = df[df.columns[2:]].values.tolist()
test_dataset = df[['comment_text', 'list']].copy()


#train_dataset=new_df.sample(frac=train_size,random_state=200)
#test_dataset=new_df.drop(train_dataset.index).reset_index(drop=True)
#train_dataset = train_dataset.reset_index(drop=True)


#train_dataset=new_df.sample(frac=train_size,random_state=200)
#test_dataset=new_df.drop(train_dataset.index).reset_index(drop=True)
#train_dataset = train_dataset.reset_index(drop=True)


#print("FULL Dataset: {}".format(new_df.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))

training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN)
testing_set = CustomDataset(test_dataset, tokenizer, MAX_LEN)

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)


class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained(args.base_model, return_dict=False)
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, label_count)

    def forward(self, ids, mask, token_type_ids):
        _, output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output


model = BERTClass()
model.to(device)

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)


def train(epoch):
    model.train()
    for _, data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.float)

        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        if _ % 5000 == 0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

for epoch in range(EPOCHS):
    train(epoch)

def validation(epoch):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets

#for epoch in range(EPOCHS):

outputs, targets = validation(epoch)
outputs = np.array(outputs) >= 0.5
accuracy = metrics.accuracy_score(targets, outputs)
f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
print(f"Accuracy Score = {accuracy}")
print(f"F1 Score (Micro) = {f1_score_micro}")
print(f"F1 Score (Macro) = {f1_score_macro}")

y_inverse_tups = multibinarizer.inverse_transform(outputs)
pred_icd_codes = []
for y_tup in y_inverse_tups:
    pred_icd_codes.append(list(y_tup))

#we have data here

testing_results = []

# load instruction data
f = open(args.test_dataset_path)
testing_data = json.load(f)

for idx, case in enumerate(testing_data):

    # extract case data
    case_text = None
    for conversation in case['conversations']:
        if conversation['from'] == 'human':
            case_text = conversation['value']

    # put in the same format as original
    icd_output = None
    for icd_code in pred_icd_codes[idx]:
        if icd_output is None:
            icd_output = icd_code + '\n'
        else:
            icd_output = icd_output + icd_code + '\n'
    if icd_output is not None:
        icd_output = icd_output.strip()

    # create test section if needed
    if 'test_results' not in case:
        case['test_results'] = dict()

    # prepare results
    result = dict()
    result['pred_icd_codes'] = icd_output

    case['test_results'][model_name] = result
    testing_results.append(case)

# rewrite original file
json_object = json.dumps(testing_results, indent=4)
with open(args.test_dataset_path, "w") as outfile:
    outfile.write(json_object)
