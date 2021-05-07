import global_var
import csv
import numpy as np # numpy==1.19.2
import math
import random
from anytree.exporter import DotExporter # anytree==2.8.0

# We use "target feature" and "label" interchangeably.

def read_data(file_path):
    """
    Reads data from file_path, 

    Constructs index_label_dict (defined in global_var.py). 
    Create a sorted list of unique label values.
    Each key is the index of the label in the sorted unique label list,  
    The value is the label value.

    To create index_label_dict, we transform the label values 
    to their indices in the sorted unique label list. 
    When plotting the tree, we transform the label indices 
    back into the label values using index_label_dict.

    :Example: Consider a list of labels ['2','4','2'].
    The sorted list of unique labels are ['2', '4']. 
    The index_label_dict is {0:'2', 1:'4'}.
    The transformed labels are [0, 1, 0]. 

    Constructs index_dict (defined in global_var.py).
    The keys are the feature names + '_index'.
    The value is the index of the feature in the csv file.
    Stores the label name in the label_name variable (defined in global_var.py).

    :Example: An index_dict is 
    {'a_index':0, 'b_index':1, 'quality_index':2}.
    feature 'a' is in the 0th column.
    feature 'b' is in the 1st column.
    the label 'quality' is in the 2nd column.
    We assume that the label column is the last column.

    :param file_path: The name of the given file.
    :type filename: str

    :return: A 2d data array consisting of examples with label indices
    (Each row is an example. Each column in an example is a feature value.),
    a list of input feature names, 
    and the number of possible label values.
    :rtype: List[List[Any]], List[str], int
    """

    data_array = []
    with open(file_path, 'r') as csv_file:

        # read csv_file into a 2d array
        reader = csv.reader(csv_file)
        for row in reader:
            data_array.append(row)

        # extract feature names
        feature_names = data_array[0]
        input_feature_names = feature_names[:-1]
        num_features = len(feature_names)

        data_array = data_array[1:] # exclude feature name row

        # a sorted list of unique labels
        labels = sorted(list(set(np.array(data_array)[:,-1])))

        # change the label values to an int, 
        # which is the index of the label in the sorted list of unique labels
        for i in range(len(data_array)):
            data_array[i][-1] = labels.index(data_array[i][-1])

        # change the input feature values to floats
        for e in data_array:
            for k in range(num_features - 1): # -1 to exclude the label column
                if e[k] == '':
                    continue
                e[k] = float(e[k])

        # create index_label_dict (defined in global_var.py)
        for i in range(len(labels)):
            global_var.index_label_dict[i] = labels[i]

        # print("index_label_dict is:", global_var.index_label_dict)

        # create index_dict (defined in global_var.py)
        for i in range(num_features):
            global_var.index_dict[feature_names[i] + "_index"] = i
            if (i + 1) == num_features: # assume label is the last column
                global_var.label_name = feature_names[i] + "_index"

        # print("index_dict is:", global_var.index_dict)

        return data_array, input_feature_names, len(labels)

    
def plot_tree(tree, file_path):
    """
    Saves a plot of the tree to file_path.
    The label values are obtained using index_label_dict.
    
    :param tree: the root node of a decision tree.
    :type tree: Node
    :param file_path: the file path to save the plot
    :type file_path: str 

    Each function requires a Node to have the following attributes.
    :attr fd: feature (on non-leaf levels) or decision (on leaves)
    :type fd: str or int
    :attr leaf: True if the Node is a leaf and False otherwise.
    :type leaf: bool 
    :attr edge: the information to be displayed on the edge between parant and child.
    That is, <= or > and split point values
    :type edge: str
    """ 

    DotExporter(tree,
                nodeattrfunc=lambda n: \
                    'label="{}"'.format(global_var.index_label_dict[n.fd]) \
                        if n.leaf else 'label="{}"'.format(n.fd),
                edgeattrfunc=lambda p, \
                    c: 'label="{}"'.format(c.edge)).to_picture(file_path)



def preprocess(data_array, folds_num=10):
    """
    Divides data_array into folds_num sets for cross validation. 
    Each fold has an approximately equal number of examples.

    :param data_array: the 2d data array
    :type data_array: List[List[Any]]
    :param folds_num: the number of folds
    :type folds_num: int, default 10
    :return: a list of sets of length folds_num
    Each set contains the set of data for the corrresponding fold.
    :rtype: List[List[List[Any]]]
    """  
    
    subset_size = math.floor(len(data_array) / folds_num)
    ret = []
    for i in range(folds_num):
        
        # print(i * subset_size, (i + 1) * subset_size)
        
        if (i + 1) == folds_num:
            ret.append(data_array[i * subset_size:])
        else:
            ret.append(data_array[i * subset_size: (i + 1) * subset_size])
    return ret


def get_depth(root):
    """
    Takes a tree with the given root
    and returns the maximum depth of the tree.

    :param root: the root node of the tree.
    :type root: Node
    :return: the max depth of the tree
    :rtype: int
    """ 

    if root.leaf:
        return 1
    else:
        return max(get_depth(root.children[0]) + 1, \
                   get_depth(root.children[1]) + 1)

