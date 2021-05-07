import math
from anytree import AnyNode, Node, RenderTree
import global_var # global variables in dt_io.py
from math import log
import numpy as np
from operator import itemgetter
from collections import Counter 

def entropy(dist):

    ent = []
    for i in dist: 
        if i == 0:
            continue
        elif i == 1:
            return 0
        else: 
            x = i * math.log(i,2)
            ent.append(x)
    return(-sum(ent))

def get_splits(examples, feature):
    index = global_var.index_dict[feature + "_index"]  
    label_index = len(global_var.index_dict) - 1 
    #sorted_examples = sorted(examples, key = itemgetter(-1))
    sorted_examples = sorted(examples, key = itemgetter(index))
    current = sorted_examples[0][index]
    currentLabel = sorted_examples[0][-1]
    classList = [0] * len(global_var.index_label_dict.keys())
    midwayList = {}
    for i in range(len(sorted_examples)-1):
        cl = classList.copy()
        if (sorted_examples[i][index] != sorted_examples[i+1][index]) and (sorted_examples[i][-1] != sorted_examples[i+1][-1]):
            sumoftwo = sorted_examples[i][index] + sorted_examples[i+1][index]
            mid = sumoftwo/2
            for j in range((i+1)):
                if sorted_examples[j][index] <= mid:
                    cl[sorted_examples[j][-1]]+=1
                midwayList[mid] = cl
    return midwayList

def choose_feature_split(examples, features):

    tlist = list(zip(*examples)) # example is list of list 
    features_list = list(tlist[-1])
    count = np.bincount(features_list)
    total_keys = len(global_var.index_label_dict.keys())
    all_feature_count = np.pad(count,(0,total_keys-len(count)), mode='constant')
    numItem = len(examples)
    ent_dict ={}
    ent_split = {}
    for f in features:
        index = features.index(f)
        examples = sorted(examples, key = itemgetter(index))
        splits = get_splits(examples,f)
        vl = []
        sp = []
        for key, value in splits.items():
            vl.append(value)
            sp.append(key)

        if len(sp) == 0:
            continue

        entropy_d ={}
        for item in sp:
            top1 = np.array(splits[item])
            denom1= sum(np.array(splits[item]))
            val = (top1/denom1).tolist()

            remain_val = all_feature_count - splits[item]
            top2 = np.array(remain_val)
            denom2 = sum(np.array(remain_val))
            remain_val_ans = (top2/denom2).tolist()

            ent1 = entropy(val)
            ent2 = entropy(remain_val_ans)
            left = ent1 * (sum(np.array(splits[item]))/numItem)
            right = ent2 * (sum(np.array(remain_val))/numItem)
            total_ent = left+right
            entropy_d[item]=total_ent

        best_split_entropy = min(entropy_d.keys(), key=(lambda k: entropy_d[k]))
        ent_dict[f] = entropy_d[best_split_entropy]
        ent_split[f] = best_split_entropy

    if len(ent_dict) == 0:
        return None, -1
    else:
        bestEnt = min(ent_dict.keys(), key=(lambda k: ent_dict[k]))
        bestSplit = ent_split[bestEnt]
    
    return bestEnt, bestSplit

def split_examples(examples, feature, split):
    """
    Splits examples into two sets by a feature and a split value.
    Returns two sets of examples.
    The first set has examples where feature value <= split value
    The second set has examples where feature value is > split value.  

    Used in split_node.

    :param examples: the 2d data array.
    :type examples: List[List[Any]]
    :param feature: the feature name
    :type feature: str
    :param split: the split value
    :type split: float

    :return: two sets of examples 
    :rtype: List[List[Any]], List[List[Any]]
    """ 

    indec_dict_list = []
    for i in (list(global_var.index_dict.keys())):
        indec_dict_list.append(i[:-6])
    features = indec_dict_list[:-1] #['sepal_len_index', 'sepal_width_index', 'petal_len_index', 'petal_width_index']
    index_of_feature = features.index(feature)
    # sort the examples with the choosen feature 
    sorted_examples = sorted(examples, key = itemgetter(index_of_feature))
    count = 0
    for i in sorted_examples:
        if i[index_of_feature] <= split:
            count += 1
    return sorted_examples[:count], sorted_examples[count:]

def split_node(cur_node, examples, features, max_depth):
    # BASE 1:  if (all examples are in the same label)   Check all examples[-1]
    indec_dict_list = []
    for i in (list(global_var.index_dict.keys())):
        indec_dict_list.append(i[:-6])
    features = indec_dict_list[:-1]
    cur_node_class = [item[-1] for item in examples] 
    #print("cur_node_class", cur_node_class)
    depth = cur_node.d
    if cur_node_class.count(cur_node_class[0]) == len(cur_node_class):   #Check if all cur_node_class are same 
        cur_node.leaf = True
        cur_node.fd = cur_node_class[0]
        return cur_node
    # BASE 2:  if (no feature left) return Majority Decision
    elif len(features) == 0:
        print("THERE IS NO FEATURE LEFT")
        #for i in cur_node_class: 
    # BASE 3:  if (no examples left) return Majority decision of parent
    elif len(examples) == 0:
        parent_node = cur_node.parent
        cur_node.majority = parent_node.majority
        return cur_node
    # BASE 4: if (cur_node_depth > max_depth) return cur_node
    elif depth == max_depth:
        cur_node.leaf = True
        occurence_count = Counter(cur_node_class)
        cur_node.fd = occurence_count.most_common(1)[0][0]
        return cur_node
    else:
        left, right = split_examples(examples, cur_node.fd, cur_node.split)
        occurence_count = Counter(cur_node_class)
        cur_majority = occurence_count.most_common(1)[0][0]
        # Get left entropy and right entropy 
        tlist_left = list(zip(*left)) # example is list of list 
        left_entropy_list = list(tlist_left[-1])
        cur_node_left = [item[-1] for item in left] 
        cur_node_right = [item[-1] for item in right] 
        occurence_count1 = Counter(cur_node_left)
        left_majority = occurence_count1.most_common(1)[0][0]
        occurence_count2 = Counter(cur_node_right)
        right_majority = occurence_count2.most_common(1)[0][0]
        if sum(left_entropy_list) == 0:
            left_ent = 0
            left_gain = 0
        else:
            left_norm_entropy_list = [float(i)/sum(left_entropy_list) for i in left_entropy_list]
            left_ent = entropy(left_norm_entropy_list)
            left_gain = cur_node.entropy - left_ent

        tlist_right = list(zip(*right)) # example is list of list 
        right_entropy_list = list(tlist_right[-1])
        if sum(right_entropy_list) == 0:
            right_ent = 0
            right_gain = 0
        else:
            right_norm_entropy_list = [float(i)/sum(right_entropy_list) for i in right_entropy_list]
            right_ent = entropy(right_norm_entropy_list)
            right_gain = cur_node.entropy - right_ent
        # Get fd of left and right
        #print("split_node", features)
        featureLeft, splitLeft = choose_feature_split(left, features)
        featureRight, splitRight = choose_feature_split(right, features)
    if depth <= max_depth:
        if len(left) < 2:
            leftNode = Node(name = cur_node.name+"leftchild"+str(depth), parent = cur_node, fd = cur_majority, leaf = True, edge = "<= "+str(cur_node.split), majority = cur_majority, split = splitLeft, d = depth + 1, entropy = left_ent, entropy_gain = left_gain)
            rightNode = Node(name = cur_node.name+"rightchild"+str(depth), parent = cur_node, fd = cur_majority, leaf = True, edge = "> "+str(cur_node.split), majority = cur_majority, split = splitRight, d = depth + 1, entropy = right_ent, entropy_gain = right_gain)
        elif len(right) < 2 :
            leftNode = Node(name = cur_node.name+"leftchild"+str(depth), parent = cur_node, fd = cur_majority, leaf = True, edge = "<= "+str(cur_node.split), majority = cur_majority, split = splitLeft, d = depth + 1, entropy = left_ent, entropy_gain = left_gain)
            rightNode = Node(name = cur_node.name+"rightchild"+str(depth), parent = cur_node, fd = cur_majority, leaf = True, edge = "> "+str(cur_node.split), majority = cur_majority, split = splitRight, d = depth + 1, entropy = right_ent, entropy_gain = right_gain)
        else:
            leftNode = Node(name = cur_node.name+"leftchild"+str(depth), parent = cur_node, fd = featureLeft, leaf = False, edge = "<= "+str(cur_node.split), majority = cur_majority, split = splitLeft, d = depth + 1, entropy = left_ent, entropy_gain = left_gain)
            rightNode = Node(name = cur_node.name+"rightchild"+str(depth), parent = cur_node, fd = featureRight, leaf = False, edge = "> "+str(cur_node.split), majority = cur_majority, split = splitRight, d = depth + 1, entropy = right_ent, entropy_gain = right_gain)
            split_node(leftNode, left, features, max_depth)
            split_node(rightNode, right, features, max_depth)

def learn_dt(examples, features, label_dim, max_depth=math.inf):
    feature, split = choose_feature_split(examples,features)
    # get entropy of root node 
    tlist = list(zip(*examples)) # example is list of list 
    entropy_list = list(tlist[-1])
    norm_entropy_list = [float(i)/sum(entropy_list) for i in entropy_list]
    ent = entropy(norm_entropy_list)
    # root node has depth = 1, feature is from choose_feature_split
    rootNode = Node(name = "Root node", parent = None, fd = feature, leaf = False, edge = " ", majority = None, split = split, d = 1, entropy = ent, entropy_gain = 0)
    #print("learn_dt", features)
    split_node(rootNode, examples, features, max_depth)
    return rootNode

def predict(tree, example, max_depth=math.inf):
    # tree is a node
    # 
    #If we haven't reached a leaf node at the max depth, 
    #we will return the majority decision at the last node
    #Base case 1: return majority decision at max depth
    if tree.d == max_depth:
        return tree.majority
    elif tree.leaf:
        return tree.fd
    else: 
        # Get the feature index
        indec_dict_list = []
        for i in (list(global_var.index_dict.keys())):
            x = i.split("_")
            indec_dict_list.append(x[0])
        features = indec_dict_list[:-1]
        index_feature = features.index(tree.fd)
        if example[index_feature] > tree.split:
            return predict(tree.children[1], example, max_depth)
        else: 
            return predict(tree.children[0], example, max_depth)


def get_prediction_accuracy(tree, examples, max_depth=math.inf):
    """
    Calculates the prediction accuracy for the given examples 
    based on the given decision tree up to the max_depth. 

    If we have not reached a leaf node at max_depth, 
    return the majority decision at the node.

    Used in cv.py.

    :param tree: a decision tree
    :type tree: Node
    :param examples: a 2d data array containing set of examples.
    :type examples: List[List[Any]]
    :param max_depth: the max depth
    :type param max_depth: int, default math.inf
    :return: the prediction accuracy for the examples based on the tree
    :rtype: float
    """ 
    correct_predict = 0
    for i in examples:
        #get prediction
        predict_label = predict(tree, i, max_depth)
        if (predict_label+3) == i[-1]:
            correct_predict += 1
    acc = correct_predict / len(examples)
    return acc



