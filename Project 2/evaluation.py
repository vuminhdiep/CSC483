"""Evaluation Metrics

Author: Kristina Striegnitz and Diep (Emma) Vu

As a student at Union College, I am part of a community that values intellectual effort, curiosity and discovery. I understand that in order to truly claim my educational and academic achievements, I am obligated to act with academic integrity. Therefore, I affirm that I will carry out my academic endeavors with full academic honesty, and I rely on my fellow students to do the same.

Complete this file for part 1 of the project.
"""

def get_confusion_matrix(y_pred, y_true):
    """Calculate the confusion matrix of the predicted labels.
    y_pred: list predicted labels
    y_true: list of corresponding true labels
    """
    if len(y_pred) != len(y_true):
        return -1
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0
    for i in range(len(y_pred)):
        if y_true[i] == 1:
            if y_pred[i] == 1:
                true_pos += 1
            else:
                false_neg += 1
        else:
            if y_pred[i] == 1:
                false_pos += 1
            else:
                true_neg += 1
    return true_pos, false_pos, true_neg, false_neg

def get_accuracy(y_pred, y_true):
    """Calculate the accuracy of the predicted labels.
    y_pred: list predicted labels
    y_true: list of corresponding true labels
    """
    true_pos, false_pos, true_neg, false_neg = get_confusion_matrix(y_pred, y_true)
    if true_pos + false_pos + true_neg + false_neg == 0:
        return -1
    accuracy = (true_pos + true_neg)/(true_pos + false_pos + true_neg + false_neg)
    return accuracy

def get_precision(y_pred, y_true):
    """Calculate the precision of the predicted labels.
    y_pred: list predicted labels
    y_true: list of corresponding true labels
    """
    true_pos, false_pos, true_neg, false_neg = get_confusion_matrix(y_pred, y_true)
    if true_pos + false_pos == 0:
        return -1
    precision = true_pos/(true_pos + false_pos)
    return precision


def get_recall(y_pred, y_true):
    """Calculate the recall of the predicted labels.
    y_pred: list predicted labels
    y_true: list of corresponding true labels
    """
    true_pos, false_pos, true_neg, false_neg = get_confusion_matrix(y_pred, y_true)
    if true_pos + false_neg == 0:
        return -1
    recall = true_pos/(true_pos + false_neg)
    return recall


def get_fscore(y_pred, y_true):
    """Calculate the f-score of the predicted labels.
    y_pred: list predicted labels
    y_true: list of corresponding true labels
    """
    precision = get_precision(y_pred, y_true)
    recall = get_recall(y_pred, y_true)
    if precision + recall == 0:
        return -1
    fscore = (2 * precision * recall)/(precision + recall)
    return fscore


def evaluate(y_pred, y_true):
    """Calculate precision, recall, and f-score of the predicted labels
    and print out the results.
    y_pred: list predicted labels
    y_true: list of corresponding true labels
    """
    accuracy = get_accuracy(y_pred, y_true)
    precision = get_precision(y_pred, y_true)
    recall = get_recall(y_pred, y_true)
    fscore = get_fscore(y_pred, y_true)
    
    print("Accuracy: {:.4%}".format(accuracy))
    print("Precision: {:.4%}".format(precision))
    print("Recall: {:.4%}".format(recall))
    print("F-score: {:.4%}".format(fscore))

if __name__ == "__main__":
    y_pred = [1,1,0,0,1,0,0,1,0,0]
    y_true = [1,1,1,1,1,0,0,0,0,0]
    
    evaluate(y_pred, y_true)


