from utils.eval import getAccuracy,getRecall,getSpanAccuracy
import json
import jieba
from fuzzywuzzy import fuzz
from utils.util import read_json_by_line
def tokenize(text, max_length=None, truncation=True):
    # Split the text into chunks using Chinese characters as separators
    chunks = []
    current_chunk = ""
    
    for char in text:
        if ord(char) < 128:
            current_chunk += char
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = ""
            chunks.append(char)
    
    if current_chunk:
        chunks.append(current_chunk)
    return chunks
# 定义函数，将字符串分词并返回分词后的列表
def tokenize_and_sort(input_string):
    tokens = jieba.lcut(input_string)
    # 将分词结果按顺序拼接成一个字符串并返回
    return " ".join(tokens)

def get_substrings_within_length_difference(input_text, specified_length):
    substrings = []
    
    for start in range(len(input_text)):
        for end in range(start + specified_length - 6, start + specified_length + 7):
            if end <= len(input_text):
                substring = input_text[start:end]
                substrings.append(substring)
    
    return substrings

def extract_CondandOp(input_str):
    lines = input_str.split('\n')
    conditions = []
    operations = []
    current_condition = ""

    for i in range(len(lines) - 1):
        line = lines[i]
        next_line = lines[i + 1]

        stripped_line = line.strip()
        next_stripped_line = next_line.strip()

        leading_spaces = len(line) - len(stripped_line)
        next_leading_spaces = len(next_line) - len(next_stripped_line)
        #print(str(leading_spaces)+" "+str(next_leading_spaces))
        # 按照空格数匹配
        if leading_spaces < next_leading_spaces:
            conditions.append(stripped_line.lstrip("if 如果 elif").rstrip(":").lstrip("在"))
        else:   
            operations.append(stripped_line)
        # 按照:匹配
        # if stripped_line.endswith(":"):
        #     conditions.append(stripped_line.lstrip("if 如果 elif").rstrip(":").lstrip("在"))
        # else:   
        #     operations.append(stripped_line)        
        if i == len(lines)-2:
            operations.append(next_stripped_line)
    return conditions,operations
def get_subscores(test_string,substrings):
    scores = []
    score = 0
    tokenized_span =jieba.lcut(test_string)
    # 输入两个字符串
    for i in substrings:

        # 分词和排序
        tokenized_string = jieba.lcut(i)

        # 使用fuzzywuzzy计算字符串相似度
        similarity_score = fuzz.ratio(tokenized_span,tokenized_string)

        scores.append(similarity_score)
    return scores
def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]
def get_levensubscores(test_string, substrings):
    scores = []
    for i in substrings:
        # 计算编辑距离
        edit_distance = levenshtein_distance(test_string, i)
        
        # 计算相似度分数，可以根据需要进行缩放或变换
        similarity_score = 1 / (1 + edit_distance)
        
        scores.append(similarity_score)
    return scores



def get_bestmatch(span,text):
    substrings= get_substrings_within_length_difference(text,len(span))
    scores = get_levensubscores(span,substrings)
    string_scores = zip(substrings,scores)
    string_scores = sorted(string_scores,key=lambda x:x[1],reverse=True)
    start_index = text.find(string_scores[0][0])
    end_index = start_index+len(string_scores[0][0])
    return string_scores[0][0],(start_index,end_index)
def get_revised_span(spans,text):
    revised=[]
    for span in spans:
        revised_span=get_bestmatch(span,text)
        revised.append(revised_span)
    return revised
# Example usage:
input_str = """
if x > 5:
    print("x is greater than 5")
elif x == 5:
    print("x is equal to 5")
else:
    print("x is less than 5")
"""

datas=read_json_by_line("./outputs/t2f_revised.json")

    
processes = []


with open("./datasets/ICTPE_v2/ICTPE_v2_editbefore.json") as f:
    gdata =json.load(f)
# gdata = [d for d in gdata if d["text"].count("前")==0]
# process_texts = []
# for i,d in enumerate(gdata):
#     if d["text"].count("前")==0:
#         process_texts.append(d["text"])
# for d in datas:
#     if d["text"].count("前")==0:
#         processes.append(d["response"])

process_texts = []
for i,d in enumerate(gdata):

    process_texts.append(d["text"])
for d in datas:

    processes.append(d["response"])
print(len(processes))
print(len(process_texts))
conditions = []
operations = []
for process in processes:
    cond,op =extract_CondandOp(process)
    conditions.append(cond)
    operations.append(op)
revised_operations = []
revised_conditions = []
for i,op in enumerate(conditions):
    revised_span=get_revised_span(op,process_texts[i])
    revised_conditions.append(revised_span)
for i,op in enumerate(operations):
    revised_span=get_revised_span(op,process_texts[i])
    revised_operations.append(revised_span)
# print(conditions)
# print(operations)


g_list=[]
g_op_list = []
g_list_with_span = []
g_op_list_with_span = []
for process in gdata:
    conds_with_span = []
    ops_with_span = []
    conds = []
    ops = []
    for node in process["node_list"]:
        if node["type"]=="condition":
            #conds.append(node["text_span"].lstrip("当 如果").rstrip("时 后 之后").replace("如果",""))
            conds_with_span.append(((node["text_span"]),(node["start"],node["end"])))
            conds.append(node["text_span"].lstrip("当 如果").rstrip("时 后 之后").replace("如果",""))
        elif node["type"]=="operate":
            ops_with_span.append((node["text_span"],(node["start"],node["end"])))
            ops.append(node["text_span"])
    g_list_with_span.append(conds_with_span)
    g_op_list_with_span.append(ops_with_span)
    g_list.append(conds)
    g_op_list.append(ops)
#print(g_list)
acc = getSpanAccuracy(g_list_with_span,revised_conditions)
print(acc)
acc = getSpanAccuracy(g_op_list_with_span,revised_operations)
print(acc)
acc=getAccuracy(conditions,g_list)
rec=getRecall(conditions,g_list)
print(str(acc)+" "+str(rec))
# # acc=getAccuracy(revised_conditions,g_list)
# # rec=getRecall(revised_conditions,g_list)
# print(str(acc)+" "+str(rec))
acc=getAccuracy(operations,g_op_list)
rec=getRecall(operations,g_op_list)
print(str(acc)+" "+str(rec))
# # acc=getAccuracy(revised_operations,g_op_list)
# # rec=getRecall(revised_operations,g_op_list)
# #print(revised_operations)
# print(str(acc)+" "+str(rec))


