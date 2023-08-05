import argparse
import json
import pandas as pd

from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer

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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Path LLM Metrics')
    parser.add_argument('--results_dataset', type=str, default='results_dataset.json', help='location of dataset')
    parser.add_argument('--results_dataset_metrics', type=str, default='result_dataset_metrics.json', help='location of dataset')
    args = parser.parse_args()

    multibinarizer_map = dict()
    y_true_onehot_map = dict()

    output_metrics = dict()

    # load instruction data
    f = open(args.results_dataset)
    testing_data = json.load(f)

    count = 0
    #loop through datasets (tiny-v1, smallv1, etc.) first
    for dataset_id, dataset_results in testing_data.items():
        #map to keep y_true for specific dataset
        y_true = []
        #pred map per model per dataset
        pred_map = dict()

        if dataset_id not in output_metrics:
            output_metrics[dataset_id] = dict()

        #loop through cases in the dataset and build comparisons for analysis
        #first loop populates datasets needed for analysis
        #this is done here so that hot encoding can see all possible data
        for case_id, case_data in dataset_results.items():

            #generate true list
            true_icd_codes = case_data['true_icd_codes']
            true_icd_codes = get_sorted_icd_list(true_icd_codes)
            y_true.append(true_icd_codes)

            #generate model list
            for model_id, model_result in case_data['model_preds'].items():
                if model_id not in pred_map:
                    pred_map[model_id] = []
                pred_icd_code = case_data['model_preds'][model_id]
                if pred_icd_code is None:
                    pred_icd_code = []
                else:
                    pred_icd_code = get_sorted_icd_list(pred_icd_code)
                pred_map[model_id].append(pred_icd_code)

        #now that we have datasets for analysis, loop through models and calc metrics

        for model_id, y_pred in pred_map.items():

        # calculate model scores

            if dataset_id not in multibinarizer_map:
                multibinarizer = MultiLabelBinarizer()
                y_true_onehot = multibinarizer.fit_transform(y_true)
                multibinarizer_map[dataset_id] = multibinarizer
                y_true_onehot_map[dataset_id] = y_true_onehot

            #pull one_hot true for dataset
            y_true_onehot = y_true_onehot_map[dataset_id]
            #calculate onehot for pred
            y_pred_onehot = multibinarizer_map[dataset_id].transform(y_pred)

            #calculate metrics
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

            metrics = dict()
            metrics['acc'] = accuracy
            metrics['auc'] = roc_auc
            metrics['precision'] = precision
            metrics['recall'] = recall
            metrics['f1'] = f1
            output_metrics[dataset_id][model_id] = metrics

    print('Creating JSON results:', args.results_dataset_metrics)
    #save results
    json_object = json.dumps(output_metrics, indent=4)
    with open(args.results_dataset_metrics, "w") as outfile:
        outfile.write(json_object)

    #create csv
    csv_output_path = args.results_dataset_metrics.replace('.json','.csv')
    metric_lists = []
    # load instruction data
    f = open(args.results_dataset_metrics)
    results_dataset_metrics = json.load(f)

    for dataset_id, dataset_result in results_dataset_metrics.items():
        for model_id, model_result in dataset_result.items():
            record = []
            record.append(dataset_id)
            record.append(model_id)
            record.append(model_result['acc'])
            record.append(model_result['auc'])
            record.append(model_result['precision'])
            record.append(model_result['recall'])
            record.append(model_result['f1'])
            metric_lists.append(record)

    df = pd.DataFrame(metric_lists, columns=['dataset_id', 'model_id', 'acc', 'auc', 'precision', 'recall', 'f1'])
    print('Creating CSV results:', csv_output_path)
    df.to_csv(args.results_dataset_metrics.replace('.json','.csv'), index=False)
