import json
with open("./datasets/ICTPE_v2/selected.json",encoding="utf-8") as f:
    procedures = json.load(f)
for p in procedures:
    op,st,cond,check = 0,0,0,0
    nodes = p["node_list"]
    for i,step in enumerate(nodes):
        if step["type"] == "operate":
            step["type_id"] ="操作"+str(op)
            op+=1
        if step["type"] == "status":
            step["type_id"] ="状态"+str(st)
            st+=1
        if step["type"] == "condition":
            step["type_id"] ="条件"+str(cond)
            cond+=1        
        if step["type"] == "check":
            step["type_id"] ="判断"+str(check)
            check+=1    
with open("./datasets/ICTPE_v2/selected_utypeid.json","w",encoding="utf-8") as f1:
    json.dump(procedures,f1,ensure_ascii=False)