import sklearn.metrics
from icecream import ic 
import json 
import argparse 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #set_seed(42)

    parser.add_argument('--file_path', help='number of epochs')

    args = parser.parse_args() 

    # labels_list = [1,0,0,1]
    # probs_list = [0.8,0.3,0.2,0.3]
    FILE_PATH = args.file_path

    with open(FILE_PATH) as json_file:
        auroc_dict = json.load(json_file)

    train_labels = auroc_dict['train_labels']
    train_probs =  auroc_dict['train_probs']

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true = train_labels, y_score = train_probs , pos_label = 1) #positive class is 1; negative class is 0
    train_auroc = sklearn.metrics.auc(fpr, tpr)
    ic(train_auroc)


    val_labels = auroc_dict['val_labels']
    val_probs =  auroc_dict['val_probs']

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true = val_labels, y_score = val_probs , pos_label = 1) #positive class is 1; negative class is 0
    val_auroc = sklearn.metrics.auc(fpr, tpr)
    ic(val_auroc)
