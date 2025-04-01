import csv
import io
import random
import sys


def writeToCsv(path, content, labels):
    """write a given content in a csv file.
    arguments :
        `path` : (string)
            the path where we will save the csv file (relative or absolute path).
        `content` : (ndarray/list)
            the content to save in the csv file.
        `labels` : (ndarray/list)
            the labels of the csv columns.
    """
    with io.open(path, "w") as fd:
        writer = csv.writer(fd)
        if labels is not None:
            content.insert(0, labels)
        writer.writerows(content)


def splitList(content, cut):
    """splits an array in two.
    arguments :
        `content` : (ndarray/list)
            the array to split in two.
        `cut` : (float) in [0:1]
            the ratio by which we will split the array.
    """
    c = int(len(content) * cut)
    return (content[:c], content[c:])


def splitDataset(path, cut=0.2, label=False, shuffle=False):
    """loads the dataset and splits it in two sets.
    arguments :
        `path` : (string)
            the path to the dataset to load (relative or absolute path).
        `cut` : (float) in [0:1]
            the ratio for the reparition of the two sets (0.2 means 20 percents
            for the test set and 80 percents for the training set).
        `label` : (boolean)
            wether the first line of the csv contains labels or not
        `shuffle` : (boolean)
            wether we shuffle the dataset before the split operation or not
    """
    labels = None
    content = []
    csvfile = open(path, "rb") if sys.version_info[0] < 3 else open(path, "rt", encoding="utf-8")
    reader = csv.reader(csvfile, delimiter=",")
    for i, row in enumerate(reader):
        if i == 0 and label is True:
            labels = row
            continue
        content.append(row)
    csvfile.close()
    if shuffle is True:
        random.shuffle(content)
    filename = path[: path.rfind(".")]
    testset, trainingset = splitList(content, float(cut))
    writeToCsv(filename + "_test.csv", testset, labels)
    writeToCsv(filename + "_train.csv", trainingset, labels)


if __name__ == "__main__":
    filepath = "./data.csv"
    splitDataset(filepath, cut=0.25, label=False, shuffle=True)
