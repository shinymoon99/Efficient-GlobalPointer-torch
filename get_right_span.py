import json
def getSpan(substring, text):
    start_index = text.find(substring)
    
    if start_index != -1:
        end_index = start_index + len(substring)
        return start_index, end_index
    else:
        return None
with open("./datasets/ICTPE_v2/ICTPE_v2 copy.json","r",encoding="utf-8") as f:
    data = json.load(f)
for sentence in data:
    text = sentence["text"]
    for node in sentence["node_list"]:
        print(node["text_span"])
        start,end = getSpan(node["text_span"],text)
        node["start"]=start
        node["end"]=end
        if "id" in node:
            del node["id"]       
with open("./datasets/ICTPE_v2/ICTPE_v2_editbefore.json","w",encoding="utf-8") as f1:
    json.dump(data,f1,ensure_ascii=False)