import numpy as np
from scipy.optimize import linear_sum_assignment

def span_similarity(span1, span2):
    # Define a similarity score between two spans.
    # You can use various metrics such as Jaccard similarity, overlap ratio, etc.
    # For example, you can calculate Jaccard similarity:
    intersection = len(set(span1) & set(span2))
    union = len(set(span1) | set(span2))
    return intersection / union if union > 0 else 0

def calculate_similarity_matrix(test_list, gold_list):
    # Calculate a similarity matrix where each cell (i, j) represents
    # the similarity score between test_list[i] and gold_list[j].
    similarity_matrix = np.zeros((len(test_list), len(gold_list)))
    for i, test_span in enumerate(test_list):
        for j, gold_span in enumerate(gold_list):
            similarity_matrix[i][j] = span_similarity(test_span, gold_span)
    return similarity_matrix

def find_best_matching(test_list, gold_list):
    # Example usage with different lengths:
    # test_list = [(0, 5), (10, 15), (20, 25)]
    # gold_list = [(2, 7), (11, 16)]

    # best_matching, match_rate = find_best_matching(test_list, gold_list)
    # print("Best Matching:", best_matching)
    # print("Overall Match Rate:", match_rate)


    # Calculate the similarity matrix.
    similarity_matrix = calculate_similarity_matrix(test_list, gold_list)
    
    # Use the Hungarian algorithm to find the optimal assignment that maximizes the overall match rate.
    row_indices, col_indices = linear_sum_assignment(-similarity_matrix)
    
    # Calculate the overall match rate.
    if len(test_list)>0:
        match_rate = similarity_matrix[row_indices, col_indices].sum() / len(test_list)
    else:
        match_rate = 0
    # Create a dictionary to represent the best matching.
    best_matching = {}
    for i, j in zip(row_indices, col_indices):
        if i < len(test_list) and j < len(gold_list):
            best_matching[test_list[i]] = gold_list[j]
    
    return best_matching, match_rate


def getSpanAccuracy(gold_list,test_list):
    # gold_list: [(span_text,span)]
    correct_rate = 0
    total = 0
    for pred_sent, gold_sent in zip(test_list, gold_list):
        #rate,num=calculate_span_accuracy(pred_sent,gold_sent)
        try:
            test_span = [x[1] for x in pred_sent]
            gold_span = [x[1] for x in gold_sent]
            best_match,match_rate = find_best_matching(test_span,gold_span)
        except:
            print(pred_sent)
            print(gold_sent)
        correct_rate+=match_rate*len(best_match)
        total+=len(best_match)
    if total!=0:
        accuracy = correct_rate/total
    else :
        accuracy = -1
    return accuracy   
def getAccuracy(gold_list,test_list):
    correct = 0
    total = 0
    for pred_sent, gold_sent in zip(test_list, gold_list):
        pred_set = set(pred_sent)
        gold_set = set(gold_sent)        
        correct += len(pred_set & gold_set)
        total += len(pred_set)
    if total!=0:
        accuracy = correct/total
    else :
        accuracy = -1
    return accuracy
def getRecall(gold_list, test_list):
    true_positives = 0
    false_negatives = 0

    for pred_sent, gold_sent in zip(test_list, gold_list):
        pred_set = set(pred_sent)
        gold_set = set(gold_sent)
        
        true_positives += len(pred_set & gold_set)
        false_negatives += len(gold_set - pred_set)
    
    if (true_positives + false_negatives) != 0:
        recall = true_positives / (true_positives + false_negatives)
    else:
        recall = -1

    return recall

def getALLNodeList(data_list):
    """
    :return SNodeList,CoNodeList,ONodeList,ChNodeList: 
    :rtype: [["",""],[]] 
    """
    sNodeList = []
    coNodeList = []
    oNodeList = []
    chNodeList = []
    for data in data_list:
        t_status = []
        t_cond = []
        t_operate = []
        t_check = []
        for node in data["node_list"]:
            if node["type"]=='status':
                t_status.append(node["text_span"])
            elif node["type"]=='condition':
                t_cond.append(node["text_span"])
            elif node["type"]=='operate':
                t_operate.append(node["text_span"])
            elif node["type"]=='check':
                t_check.append(node["text_span"])
        sNodeList.append(t_status)
        coNodeList.append(t_cond)
        oNodeList.append(t_operate)
        chNodeList.append(t_check)
    return sNodeList,coNodeList,oNodeList,chNodeList
def convert_edge_list_to_text_span(data_dict):
    node_dict = {node['id']: node['text_span'] for node in data_dict['node_list']}
    converted_edge_list = []

    for edge in data_dict['edge_list']:
        print(edge)
        source_id = edge[0]
        target_id = edge[1]
        source_text_span = node_dict[source_id]
        target_text_span = node_dict[target_id]
        converted_edge_list.append((source_text_span,target_text_span))

    return converted_edge_list

def getEdgeList(data_list):
    """
    :return edge_list: 
    :rtype: [["",""],[]] 
    """
    edge_list = []
    for data in data_list:
        for edge in data["edge_list"]:
            edge_list.append(edge)
    return edge_list

def getAllConvertedEdgeList(data_list):
    """
    :return edge_list: 
    :rtype: [["",""],[]] 
    """
    edge_list = []
    for data in data_list:
        edge_list.append(convert_edge_list_to_text_span(data))
    return edge_list
## main_content
# converted_edges = convert_edge_list_to_text_span(data)
# for edge in converted_edges:
#     print(edge)

