import argparse
import json
import os

import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from torch.nn import BCEWithLogitsLoss
from transformers import LongformerTokenizerFast, \
LongformerModel, LongformerConfig, Trainer, TrainingArguments, EvalPrediction, AutoTokenizer
from transformers.models.longformer.modeling_longformer import LongformerPreTrainedModel, LongformerClassificationHead
from torch.utils.data import Dataset, DataLoader
import random
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

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


parser = argparse.ArgumentParser(description='LongFormer NLP')
parser.add_argument('--train_dataset_path', type=str, default='train_dataset.json', help='location of dataset')
parser.add_argument('--test_dataset_path', type=str, default='test_dataset.json', help='location of dataset')
parser.add_argument('--num_epoch', type=int, default=1, help='location of dataset')
parser.add_argument('--cuda_device', type=int, default=0, help='location of dataset')
args = parser.parse_args()

#prepare the data
gen_bert_dataset(args.train_dataset_path, 'path_train.csv')
label_count, val_list, multibinarizer = gen_bert_dataset(args.test_dataset_path, 'path_test.csv')

# save file
base_file = os.path.basename(args.train_dataset_path)
base_filename, base_file_extension = os.path.splitext(base_file)
model_name = 'custom-longformer-149m_' + base_filename + '_epochs' + str(args.num_epoch)



# read the dataframe
#insults = pd.read_csv('data/jigsaw/train.csv')
insults = pd.read_csv('path_train.csv')

#insults = insults.iloc[0:6000]
insults['labels'] = insults[insults.columns[2:]].values.tolist()
insults = insults[['id','comment_text', 'labels']].reset_index(drop=True)

train_size = 0.9
train_dataset=insults.sample(frac=train_size,random_state=200)
test_dataset=insults.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)


# instantiate a class that will handle the data
class Data_Processing(object):
    def __init__(self, tokenizer, id_column, text_column, label_column):
        # define the text column from the dataframe
        self.text_column = text_column.tolist()

        # define the label column and transform it to list
        self.label_column = label_column

        # define the id column and transform it to list
        self.id_column = id_column.tolist()

    # iter method to get each element at the time and tokenize it using bert
    def __getitem__(self, index):
        comment_text = str(self.text_column[index])
        comment_text = " ".join(comment_text.split())

        inputs = tokenizer.encode_plus(comment_text,
                                       add_special_tokens=True,
                                       max_length=2024,
                                       padding='max_length',
                                       return_attention_mask=True,
                                       truncation=True,
                                       return_tensors='pt')
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        labels_ = torch.tensor(self.label_column[index], dtype=torch.float)
        id_ = self.id_column[index]
        return {'input_ids': input_ids[0], 'attention_mask': attention_mask[0],
                'labels': labels_, 'id_': id_}

    def __len__(self):
        return len(self.text_column)

batch_size = 4
# create a class to process the traininga and test data
tokenizer = AutoTokenizer.from_pretrained('allenai/longformer-base-4096',
                                                    padding = 'max_length',
                                                    truncation=True,
                                                    max_length = 2048)
training_data = Data_Processing(tokenizer,
                                train_dataset['id'],
                                train_dataset['comment_text'],
                                train_dataset['labels'])

test_data =  Data_Processing(tokenizer,
                             test_dataset['id'],
                             test_dataset['comment_text'],
                             test_dataset['labels'])

# use the dataloaders class to load the data
dataloaders_dict = {'train': DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=2),
                    'val': DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=2)
                   }

dataset_sizes = {'train':len(training_data),
                 'val':len(test_data)
                }

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:" + str(args.cuda_device) if torch.cuda.is_available() else "cpu")
print(device)

# check we are getting the right output
a = next(iter(dataloaders_dict['val']))
a['id_']
#len(dataloaders_dict['train'])

# instantiate a Longformer for multilabel classification class

class LongformerForMultiLabelSequenceClassification(LongformerPreTrainedModel):
    """
    We instantiate a class of LongFormer adapted for a multilabel classification task.
    This instance takes the pooled output of the LongFormer based model and passes it through a
    classification head. We replace the traditional Cross Entropy loss with a BCE loss that generate probabilities
    for all the labels that we feed into the model.
    """

    def __init__(self, config, pos_weight=None):
        super(LongformerForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.pos_weight = pos_weight
        self.longformer = LongformerModel(config)
        self.classifier = LongformerClassificationHead(config)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, global_attention_mask=None,
                token_type_ids=None, position_ids=None, inputs_embeds=None,
                labels=None):

        # create global attention on sequence, and a global attention token on the `s` token
        # the equivalent of the CLS token on BERT models
        if global_attention_mask is None:
            global_attention_mask = torch.zeros_like(input_ids)
            global_attention_mask[:, 0] = 1

        # pass arguments to longformer model
        outputs = self.longformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids)

        # if specified the model can return a dict where each key corresponds to the output of a
        # LongformerPooler output class. In this case we take the last hidden state of the sequence
        # which will have the shape (batch_size, sequence_length, hidden_size).
        sequence_output = outputs['last_hidden_state']

        # pass the hidden states through the classifier to obtain thee logits
        logits = self.classifier(sequence_output)
        outputs = (logits,) + outputs[2:]
        if labels is not None:
            loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.num_labels),
                            labels.view(-1, self.num_labels))
            # outputs = (loss,) + outputs
            outputs = (loss,) + outputs

        return outputs

model = LongformerForMultiLabelSequenceClassification.from_pretrained('allenai/longformer-base-4096',
                                                  #'/media/data_files/github/website_tutorials/results/longformer_2048_multilabel_jigsaw',
                                                  gradient_checkpointing=False,
                                                  attention_window = 512,
                                                  num_labels = label_count,
                                                  cache_dir='data',
                                                                     return_dict=True)

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score


# acc = accuracy_score(labels, preds)
# acc = accuracy_score(labels, preds)

def multi_label_metric(
        predictions,
        references,
):
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    y_pred = np.zeros(probs.shape)
    y_true = references
    y_pred[np.where(probs >= 0.5)] = 1
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average='micro')
    accuracy = accuracy_score(y_true, y_pred)
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metric(
        predictions=preds,
        references=p.label_ids
    )
    return result

# define the training arguments
training_args = TrainingArguments(
    output_dir = 'results',
    num_train_epochs = args.num_epoch,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 64,
    per_device_eval_batch_size= 16,
    evaluation_strategy = "steps",
    disable_tqdm = False,
    load_best_model_at_end=True,
    warmup_steps = 1500,
    learning_rate = 2e-5,
    weight_decay=0.01,
    logging_steps = 10,
    fp16 = False,
    logging_dir='logs',
    dataloader_num_workers = 0,
    run_name = 'longformer_multilabel_paper_trainer_2048_2e5a'
)

# instantiate the trainer class and check for available devices
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=training_data,
    eval_dataset=test_data,
    compute_metrics = compute_metrics,
    #data_collator = Data_Processing(),

)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

trainer.train()

trainer.evaluate()

#insults_test = pd.read_csv('data/jigsaw/test.csv')
insults_test = pd.read_csv('path_test.csv')

# instantiate a class that will handle the data
class Data_Processing_test():
    def __init__(self, tokenizer, id_column, text_column):
        # define the text column from the dataframe
        self.text_column = text_column.tolist()

        # define the id column and transform it to list
        self.id_column = id_column.tolist()

    # iter method to get each element at the time and tokenize it using bert
    def __getitem__(self, index):
        comment_text = str(self.text_column[index])
        comment_text = " ".join(comment_text.split())

        inputs = tokenizer.encode_plus(comment_text,
                                       add_special_tokens=True,
                                       max_length=2048,
                                       padding='max_length',
                                       return_attention_mask=True,
                                       truncation=True,
                                       return_tensors='pt')
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        id_ = self.id_column[index]
        return {'input_ids': input_ids[0], 'attention_mask': attention_mask[0],
                'id_': id_}

    def __len__(self):
        return len(self.text_column)

batch_size = 16
# create a class to process the traininga and test data

test_data_pred =  Data_Processing_test(tokenizer, insults_test['id'],insults_test['comment_text'])

# use the dataloaders class to load the data
dataloaders_dict = {'test': DataLoader(test_data_pred, batch_size=batch_size, shuffle=True, num_workers=2)}

def prediction():
    prediction_data_frame_list = []
    with torch.no_grad():
        trainer.model.eval()
        for i, batch in enumerate(dataloaders_dict['test']):
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            # feed the sequences to the model, specifying the attention mask
            outputs = model(inputs, attention_mask=attention_mask)
            # feed the logits returned by the model to the softmax to classify the function
            sigmoid = torch.nn.Sigmoid()
            probs = sigmoid(torch.Tensor(outputs[0].detach().cpu().data.numpy()))
            #probs.
            probs = np.array(probs)
            #print(np.array([[i] for i in probs]))
            y_pred = np.zeros(probs.shape)
            y_pred = probs
            temp_data = pd.DataFrame(zip(batch['id_'], probs), columns = ['id', 'target'])
            #print(temp_data)
            prediction_data_frame_list.append(temp_data)

    prediction_df = pd.concat(prediction_data_frame_list)
    #"val_0","val_1","val_2","val_3","val_4","val_5","val_6","val_7"
    #prediction_df[['toxic','severe_toxic','obscene','threat','insult','identity_hate']] = pd.DataFrame(prediction_df.target.tolist(),index= prediction_df.index)
    #prediction_df[["val_0","val_1","val_2","val_3","val_4","val_5","val_6","val_7"]] = pd.DataFrame(
    prediction_df[val_list] = pd.DataFrame(prediction_df.target.tolist(), index=prediction_df.index)
    prediction_df = prediction_df.drop(columns = 'target')
    return prediction_df

predictions = prediction()

predictions.to_csv('last_results.csv', index=False)

#trainer.model.save_pretrained('/media/data_files/github/website_tutorials/results/longformer_base_multilabel_2048_2e5')
#tokenizer.save_pretrained('/media/data_files/github/website_tutorials/results/longformer_base_multilabel_2048_2e5')

y_pred_onehot = []
# Using readlines()
file1 = open('last_results.csv', 'r')
Lines = file1.readlines()

count = 0
# Strips the newline character
for line in Lines:
    count += 1
    if count > 1:
        pred = []
        for pred_score in line.strip().split(',')[1:]:
            if float(pred_score) >= 0.5:
                pred.append(1)
            else:
                pred.append(0)
        y_pred_onehot.append(pred)

y_pred_onehot = np.array(y_pred_onehot)
y_inverse_tups = multibinarizer.inverse_transform(y_pred_onehot)
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
