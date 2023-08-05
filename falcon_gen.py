import json


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


if __name__ == '__main__':

    generate_falcon_v1_dataset()