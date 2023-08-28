import json
with open("datasets/ICTPE/ICTPE_all_templated.json","r",encoding="utf-8") as f:
    procedures = json.load(f)
sequences = []
for p in procedures:
    typeid2span = {}
    text = p["text"]
    for node in p["node_list"]:
        typeid2span[node["type_id"]] = (node["start"],node["end"])
    for edge in p["edge_list"]:
         # 生成next       
        typeid = edge[0]  
        sequence = {}
        sequence["text"] = text + "[SEP]" +edge[0] +"[SEP]"
        sequence["entities"] = []
        entity = {"type":"next","start":typeid2span[edge[1]][0],"end":typeid2span[edge[1]][1],"type_id":edge[1]}
        sequence["entities"].append(entity)
        
        sequences.append(sequence)
with open("datasets/ICTES/ICTES.json","w",encoding="utf-8") as f1:
    json.dump(sequences,f1,ensure_ascii=False)