import json
import os

import numpy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter
import pandas as pd

def get_sorted_icd_list(icd_code_test):

    icd_code = icd_code_test.split('\n')
    icd_code.sort()
    return icd_code

def clean_cancer_icd(cancer_icd_list, case):

    #remove any previous testing
    if 'test_results' in case:
        del case['test_results']

    icd_code_text = case['conversations'][1]['value']
    canidate_icd_code = get_sorted_icd_list(icd_code_text)

    icd_code = []
    for code in canidate_icd_code:
        if code in cancer_icd_list:
            icd_code.append(code)

    icd_output = None
    for icd in icd_code:
        if icd_output is None:
            icd_output = icd + '\n'
        else:
            icd_output = icd_output + icd + '\n'
    icd_output = icd_output.strip()

    #if icd_code_text != icd_output:
    #    print('change: all',get_sorted_icd_list(icd_code_text),'cancer:',get_sorted_icd_list(icd_output))

    case['conversations'][1]['value'] = icd_output
    return case

if __name__ == '__main__':

    #get codes for cancer
    cancer_icd_data = pd.read_csv('cancer_icd.txt', encoding='latin-1', sep='\t')
    cancer_icd_list = []

    #for col in cancer_icd_data.columns:
    #    print(col)

    for index, row in cancer_icd_data.iterrows():
        icd_code = row['ICD-10-CM Code Specific'].split('.')[0]
        if icd_code not in cancer_icd_list:
            cancer_icd_list.append(icd_code)

    #parse data
    #raw_dataset = 'path_llm_train_small.json'
    raw_dataset = 'pathicd-large-v1.json'

    # load instruction data
    f = open(raw_dataset)
    raw_data = json.load(f)

    cleaned_data = []

    X = []
    y = []

    for case in raw_data:

        id = case['id']
        # extract real and pred data
        icd_code_text = None
        for conversation in case['conversations']:
            if conversation['from'] == 'gpt':
                icd_code_text = conversation['value']

        canidate_icd_code = get_sorted_icd_list(icd_code_text)
        icd_code = []
        for code in canidate_icd_code:
            if code in cancer_icd_list:
                icd_code.append(code)

        if len(icd_code) > 0:
            X.append(id)
            y.append(icd_code)

    multibinarizer = MultiLabelBinarizer()
    y = multibinarizer.fit_transform(y)

    label_list = [] #list of y patterns
    y_index = [] #index of pattern assoicated with y
    y_map = dict()  # map between y index and index of label_list type

    for idx, label in enumerate(y):
        label = label.tolist()
        if label not in label_list:
            label_list.append(label)
        index = label_list.index(label)
        y_index.append(index)
        y_map[idx] = index

    y = numpy.array(y_index)

    keep_list = [] #list of label_list types of keep based on index
    combination_counts = Counter(y)
    for key,value in combination_counts.items():
        if value > 10:
            keep_list.append(key)

    clean_X = []
    clean_y = []

    for y_idx, label_idx, in y_map.items():
        if label_idx in keep_list:
            clean_X.append(X[y_idx])
            clean_y.append(y[y_idx])


    clean_X = pd.DataFrame(clean_X, columns=['id']) #do this to keep cases in a readable format

    #X_train, X_test, y_train, y_test = train_test_split(clean_X, clean_y, stratify=y, test_size=0.25)
    X_train, X_test, y_train, y_test = train_test_split(clean_X, clean_y, test_size=0.1, stratify=clean_y, random_state=42)

    print('train size:', len(X_train),len(y_train))
    print('test size:', len(X_test),len(y_test))

    #confirm distrobution:
    combination_counts_train = Counter(y_train)
    combination_counts_test = Counter(y_test)

    for key, value in combination_counts_train.items():
        if key not in combination_counts_test:
            print('train key not found in test:', key, value)

    train_id_list = []
    for index, row in X_train.iterrows():
        train_id_list.append(row['id'])

    test_id_list = []
    for index, row in X_test.iterrows():
        test_id_list.append(row['id'])

    train_data = []
    test_data = []


    #create datasets
    for case in raw_data:

        if (case['id'] in train_id_list) or (case['id'] in test_id_list):
            case = clean_cancer_icd(cancer_icd_list, case)
            if case['id'] in train_id_list:
                train_data.append(case)
            else:
                test_data.append(case)

    #save file
    base_file = os.path.basename(raw_dataset)
    base_filename, base_file_extension = os.path.splitext(base_file)
    
    train_file_name = base_filename + '-train-cancer.json'
    json_object = json.dumps(train_data, indent=4)
    with open(train_file_name, "w") as outfile:
        outfile.write(json_object)

    test_file_name = base_filename + '-test-cancer.json'
    json_object = json.dumps(test_data, indent=4)
    with open(test_file_name, "w") as outfile:
        outfile.write(json_object)

