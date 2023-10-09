import json
def getSpan(substring, text):
    start_index = text.find(substring)
    
    if start_index != -1:
        end_index = start_index + len(substring)
        return start_index, end_index
    else:
        return None
with open("datasets/ICTPE_v2/ICTPE_templated.json","r",encoding="utf-8") as f:
    datas = json.load(f)
# get dict
# for p in datas:
#     id2typeid = {}
#     for node in p["node_list"]:
#         id2typeid[node["id"]] = node["type_id"] 
#     for edge in p["edge_list"]:  
#         edge[0],edge[1]=id2typeid[edge[0]],id2typeid[edge[1]]
#get the right span
for p in datas:
    for node in p["node_list"]:
        start,end = getSpan(node["type_id"],p["text"])
        node["start"] = start
        node["end"] = end
with open("datasets/ICTPE_v2/ICTPE_all_templated.json","w",encoding="utf-8") as f1:
    json.dump(datas,f1,ensure_ascii=False)