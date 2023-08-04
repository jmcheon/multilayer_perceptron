import numpy as np

def count_labels(y_true, y_pred):
    label_counts = {}
    for t, p in zip(y_true, y_pred):
        t = tuple(t)
        p = tuple(p)
        
        if t not in label_counts:
            label_counts[t] = {'true_positives': 0, 'false_positives': 0, 'false_negatives': 0}
        if p not in label_counts:
            label_counts[p] = {'true_positives': 0, 'false_positives': 0, 'false_negatives': 0}

        if t == p:
            label_counts[t]['true_positives'] += 1
        else:
            label_counts[p]['false_positives'] += 1
            label_counts[t]['false_negatives'] += 1

    return label_counts

def accuracy_score(y_true, y_pred):
    label_counts = count_labels(y_true, y_pred)
    true_positives = np.sum([count['true_positives'] for count in label_counts.values()])
    total = len(y_true)
    return np.array(true_positives / total)

def precision_score(y_true, y_pred, zero_division=0):
    label_counts = count_labels(y_true, y_pred)
    precisions = []
    for label, count in label_counts.items():
        try:
            precision = count['true_positives'] / (count['true_positives'] + count['false_positives'])
        except ZeroDivisionError:
            precision = zero_division
        precisions.append(precision)
    return np.array(np.mean(precisions))

def recall_score(y_true, y_pred):
    label_counts = count_labels(y_true, y_pred)
    recalls = []
    for label, count in label_counts.items():
        recall = count['true_positives'] / (count['true_positives'] + count['false_negatives'])
        recalls.append(recall)
    return np.array(np.mean(recalls))

def f1_score(y_true, y_pred):
    label_counts = count_labels(y_true, y_pred)
    f1_scores = []
    for label, count in label_counts.items():
        try:
            precision = count['true_positives'] / (count['true_positives'] + count['false_positives'])
            recall = count['true_positives'] / (count['true_positives'] + count['false_negatives'])
            f1_score = 2 * (precision * recall) / (precision + recall)
        except ZeroDivisionError:
            f1_score = 0
        f1_scores.append(f1_score)
    return np.array(np.mean(f1_scores))

