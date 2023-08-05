import json
import time
from statistics import mean

import requests
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

def get_request(host, port, content_text,model):
    headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
    payload = dict()
    payload['model'] = model
    payload['stream'] = False

    message = dict()
    message['role'] = 'user'
    message['content'] = content_text
    payload['messages'] = [message]

    payload['max_tokens'] = 512
    payload['n'] = 2
    payload['use_beam_search'] = True
    payload['temperature'] = 0

    response = requests.post('http://'+ host +':' + port +'/v1/chat/completions', headers=headers, json=payload).json()

    response = response['choices'][0]['message']['content']
    response = response.replace('</s>','')
    response = response.replace(' ', '')
    response = response.split('\n')
    return response

def test_model(host, port, testing_dataset, model):

    testing_results = []

    time_results = []

    # load instruction data
    f = open(testing_dataset)
    testing_data = json.load(f)

    for case in testing_data:

        #extract case data
        case_text = None
        for conversation in case['conversations']:
            if conversation['from'] == 'human':
                case_text = conversation['value']

        start = time.time()

        pred_icd_codes = get_request(host, port, case_text, model)
        # Code block to be measured

        end = time.time()
        time_results.append(end - start)
        print('mean exec: ', mean(time_results))

        # put in the same format as original
        icd_output = None
        for icd_code in pred_icd_codes:
            if icd_output is None:
                icd_output = icd_code + '\n'
            else:
                icd_output = icd_output + icd_code + '\n'
        icd_output = icd_output.strip()

        #create test section if needed
        if 'test_results' not in case:
            case['test_results'] = dict()

        #prepare results
        result = dict()
        result['pred_icd_codes'] = icd_output

        case['test_results'][model] = result
        testing_results.append(case)

    #rewrite original file
    json_object = json.dumps(testing_results, indent=4)
    with open(testing_dataset, "w") as outfile:
        outfile.write(json_object)

def get_sorted_icd_list(icd_code_test, flatten=False):

    icd_code = icd_code_test.split('\n')
    icd_code.sort()

    if flatten:
        # put in the same format as original
        icd_output = None
        for icd_code in icd_code:
            if icd_output is None:
                icd_output = icd_code + '\n'
            else:
                icd_output = icd_output + icd_code + '\n'
        if icd_output is not None:
            icd_code = icd_output.strip()
        else:
            return None

    return icd_code

def generate_falcon_v1_dataset():
    falcon_train = []

    f = open('training.json')
    testing_data = json.load(f)

    for case in testing_data:
        # extract real and pred data
        conversation = case['conversations']
        input_text = conversation[0]['value']
        output_text = get_sorted_icd_list(conversation[1]['value'])
        convo = dict()
        convo['instruction'] = input_text
        convo['input'] = ''
        convo['output'] = output_text
        falcon_train.append(convo)

    json_object = json.dumps(falcon_train, indent=4)
    with open('training_falcon.json', "w") as outfile:
        outfile.write(json_object)

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

    return len(y_true_onehot[0]), val_list, multibinarizer, y

def get_test_true(input_dataset_path):

    # load instruction data
    f = open(input_dataset_path)
    data = json.load(f)

    y = []

    for case in data:

        # extract real and pred data
        icd_code_text = None
        for conversation in case['conversations']:
            if conversation['from'] == 'gpt':
                icd_code_text = conversation['value']

        icd_code = get_sorted_icd_list(icd_code_text)
        y.append(icd_code)

    # One-hot encode the lists
    multibinarizer = MultiLabelBinarizer()

    #return multibinarizer.fit_transform(y), multibinarizer
    return multibinarizer.fit_transform(y)

def consolidate_results():
    results_dataset = dict()

    from pathlib import Path
    results = list(Path("data").rglob("*.json"))
    for result in results:
        if 'test' in str(result):
            result_s = str(result).split('-')
            dataset_size_s = result_s[1]
            dataset_version_s = result_s[2]
            key = dataset_version_s + '-' + dataset_size_s

            if key not in results_dataset:
                results_dataset[key] = dict()

            # load instruction data
            f = open(result)
            testing_data = json.load(f)

            for idx_case, case in enumerate(testing_data):
                case_id = case['id']
                # print(case_id)
                if case_id not in results_dataset[key]:
                    model_record = dict()
                    # model_record['id'] = case['id']
                    model_record['input_text_size'] = len(case['conversations'][0]['value'])
                    icd_code_text = case['conversations'][1]['value']
                    icd_code = get_sorted_icd_list(icd_code_text, flatten=True)
                    model_record['true_icd_codes'] = icd_code
                    model_record['model_preds'] = dict()
                    results_dataset[key][case_id] = model_record

                test_results = case["test_results"]

                for model_key, model_response in test_results.items():
                    # sort codes
                    pred_icd_codes = model_response['pred_icd_codes']
                    if pred_icd_codes is not None:
                        # sort
                        pred_icd_codes = get_sorted_icd_list(pred_icd_codes, flatten=True)
                        model_response['pred_icd_codes'] = pred_icd_codes

                    model_key = model_key.split('/')
                    model_key = model_key[len(model_key) - 1].split('.')[0]

                    if model_key not in results_dataset[key][case_id]['model_preds']:
                        # results_dataset[key]['model_preds'][model_key] = dict()
                        results_dataset[key][case_id]['model_preds'][model_key] = pred_icd_codes
                    else:
                        if results_dataset[key][case_id]['model_preds'][model_key] != pred_icd_codes:
                            if results_dataset[key][case_id]['model_preds'][model_key] is None:
                                results_dataset[key][case_id]['model_preds'][model_key] = pred_icd_codes
                            else:
                                # print('mismatch: existing:', results_dataset[key]['model_preds'][model_key], 'new:',pred_icd_codes)
                                true_icd_codes = results_dataset[key][case_id]['true_icd_codes']
                                if pred_icd_codes == true_icd_codes:
                                    results_dataset[key]['model_preds'][model_key] = pred_icd_codes
                                    # print('updating icd code:', pred_icd_codes, key)

    json_object = json.dumps(results_dataset, indent=4)
    with open('results_dataset.json', "w") as outfile:
        outfile.write(json_object)

def get_api_results():
    host = 'localhost'
    port = '8000'

    # tiny
    #testing_dataset = 'data/api/v1/pathicd-tiny-v1-test-cancer.json'
    # 7b
    # {'train_runtime': 27.0616, 'train_samples_per_second': 4.73, 'train_steps_per_second': 0.037, 'train_loss': 6.55072021484375, 'epoch': 1.0}
    # model = 'models/checkpoints/custom-vicuna-7b_pathicd-tiny-v1-train-cancer_epochs1-fp16'
    # {'train_runtime': 68.8736, 'train_samples_per_second': 5.575, 'train_steps_per_second': 0.044, 'train_loss': 5.138146082560222, 'epoch': 3.0}
    # model = 'models/checkpoints/custom-vicuna-7b_pathicd-tiny-v1-train-cancer_epochs3-fp16'
    # {'train_runtime': 132.88, 'train_samples_per_second': 5.78, 'train_steps_per_second': 0.045, 'train_loss': 2.845317949851354, 'epoch': 6.0}
    # model = 'models/checkpoints/custom-vicuna-7b_pathicd-tiny-v1-train-cancer_epochs6-fp16'
    # {'train_runtime': 261.0325, 'train_samples_per_second': 5.884, 'train_steps_per_second': 0.046, 'train_loss': 1.4912825956319768, 'epoch': 12.0}
    # model = 'models/checkpoints/custom-vicuna-7b_pathicd-tiny-v1-train-cancer_epochs24-fp16'
    #13b
    # {'train_runtime': 25.7806, 'train_samples_per_second': 4.965, 'train_steps_per_second': 0.039, 'train_loss': 6.55072021484375, 'epoch': 1.0}
    #model = 'models/checkpoints/custom-vicuna-13b_pathicd-tiny-v1-train-cancer_epochs1-fp16'
    # {'train_runtime': 381.8379, 'train_samples_per_second': 1.006, 'train_steps_per_second': 0.016, 'train_loss': 2.7634547352790833, 'epoch': 3.0}
    #model = 'models/checkpoints/custom-vicuna-13b_pathicd-tiny-v1-train-cancer_epochs3-fp16'
    # {'train_runtime': 715.3875, 'train_samples_per_second': 1.074, 'train_steps_per_second': 0.017, 'train_loss': 1.5439812956998746, 'epoch': 6.0}
    #model = 'models/checkpoints/custom-vicuna-13b_pathicd-tiny-v1-train-cancer_epochs6-fp16'

    # small
    #testing_dataset = 'data/api/v1/pathicd-small-v1-test-cancer.json'
    # 7b
    # {'train_runtime': 586.469, 'train_samples_per_second': 5.893, 'train_steps_per_second': 0.046, 'train_loss': 1.1190487996295646, 'epoch': 1.0}
    #MISSING##model = 'models/checkpoints/custom-vicuna-7b_pathicd-small-v1-train-cancer_epochs1-fp16'
    # {'train_runtime': 1748.243, 'train_samples_per_second': 5.931, 'train_steps_per_second': 0.046, 'train_loss': 0.4943756882423236, 'epoch': 3.0}
    # model = 'models/checkpoints/custom-vicuna-7b_pathicd-small-v1-train-cancer_epochs3-fp16'
    # {'train_runtime': 3479.8285, 'train_samples_per_second': 5.959, 'train_steps_per_second': 0.047, 'train_loss': 0.28376815576152303, 'epoch': 6.0}
    # model = 'models/checkpoints/custom-vicuna-7b_pathicd-small-v1-train-cancer_epochs6-fp16'
    # 13b
    #{'train_runtime': 2612.6559, 'train_samples_per_second': 1.323, 'train_steps_per_second': 0.021, 'train_loss': 0.7647281929298684, 'epoch': 1.0}
    #model = 'models/checkpoints/custom-vicuna-13b_pathicd-small-v1-train-cancer_epochs1-fp16'
    # {'train_runtime': 8036.9887, 'train_samples_per_second': 1.29, 'train_steps_per_second': 0.02, 'train_loss': 0.3726253565170883, 'epoch': 3.0}
    #model = 'models/checkpoints/custom-vicuna-13b_pathicd-small-v1-train-cancer_epochs3-fp16'
    #{'train_runtime': 15754.3516, 'train_samples_per_second': 1.316, 'train_steps_per_second': 0.021, 'train_loss': 0.21238457266655233, 'epoch': 6.0}
    #model = 'models/checkpoints/custom-vicuna-13b_pathicd-small-v1-train-cancer_epochs6-fp16'

    # large
    testing_dataset = 'data/api/v1/pathicd-large-v1-test-cancer.json'
    #7b
    # {'train_runtime': 6812.9705, 'train_samples_per_second': 5.968, 'train_steps_per_second': 0.047, 'train_loss': 0.3647356625611098, 'epoch': 1.0}
    # model = 'models/checkpoints/custom-vicuna-7b_pathicd-large-v1-train-cancer_epochs1-fp16'
    # {'train_runtime': 20421.1249, 'train_samples_per_second': 5.973, 'train_steps_per_second': 0.047, 'train_loss': 0.2362972479450715, 'epoch': 2.99}
    # model = 'models/checkpoints/custom-vicuna-7b_pathicd-large-v1-train-cancer_epochs3-fp16'
    # {'train_runtime': 40856.0116, 'train_samples_per_second': 5.971, 'train_steps_per_second': 0.047, 'train_loss': 0.15539160127896956, 'epoch': 5.99}
    #model = 'models/checkpoints/custom-vicuna-7b_pathicd-large-v1-train-cancer_epochs6-fp16'
    # model = 'models/checkpoints/custom-vicuna-7b_pathicd-large-v1-train-cancer_epochs12-fp16'
    #model = 'models/checkpoints/custom-vicuna-7b_pathicd-large-v1-train-cancer_epochs24-fp16'
    # 7b-4bit
    # model = '/models/custom-vicuna-7b_pathicd-large-v1-train-cancer_epochs1_ggml-model-q4_0.bin'
    model = '/models/custom-vicuna-7b_pathicd-large-v1-train-cancer_epochs6_ggml-model-q4_0.bin'
    #13b
    # model = 'models/checkpoints/custom-vicuna-13b_pathicd-large-v1-train-cancer_epochs1-fp16'
    # model = 'models/checkpoints/custom-vicuna-13b_pathicd-large-v1-train-cancer_epochs3-fp16'
    #model = 'models/checkpoints/custom-vicuna-13b_pathicd-large-v1-train-cancer_epochs6-fp16'

    test_model(host, port, testing_dataset, model)

    # load instruction data
    f = open(testing_dataset)
    testing_data = json.load(f)

    y_true = []
    y_pred = []

    for case in testing_data:

        # extract real and pred data
        icd_code_text = None
        for conversation in case['conversations']:
            if conversation['from'] == 'gpt':
                icd_code_text = conversation['value']

        icd_code = get_sorted_icd_list(icd_code_text)
        y_true.append(icd_code)

        pred_icd_code_text = case['test_results'][model]['pred_icd_codes']
        pred_icd_code = get_sorted_icd_list(pred_icd_code_text)
        y_pred.append(pred_icd_code)

    # One-hot encode the lists
    multibinarizer = MultiLabelBinarizer()

    y_true_onehot = multibinarizer.fit_transform(y_true)
    y_pred_onehot = multibinarizer.transform(y_pred)

    accuracy = accuracy_score(y_true_onehot, y_pred_onehot)
    accuracy = round(accuracy, 4)

    roc_auc = roc_auc_score(y_true_onehot, y_pred_onehot)
    roc_auc = round(roc_auc, 4)

    # Calculate the precision and recall scores
    precision = precision_score(y_true_onehot, y_pred_onehot, average='samples')
    precision = round(precision, 4)

    recall = recall_score(y_true_onehot, y_pred_onehot, average='samples')
    recall = round(recall, 4)
    # Calculate the F1 score
    f1 = f1_score(y_true_onehot, y_pred_onehot, average='samples')
    f1 = round(f1, 4)

    print('Accuracy:', accuracy)
    print('AUC:', roc_auc)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 score:', f1)

    results = dict()
    results['acc'] = accuracy
    results['precision'] = precision
    results['recall'] = recall
    results['f1'] = f1


if __name__ == '__main__':


    get_api_results()
    consolidate_results()




