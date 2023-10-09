import json
def replace_substrings(original_text, replacements):
    new_text = ""
    current_index = 0
    length_difference = 0
    sorted_replacements = sorted(replacements, key=lambda x: x[0])
    for start, end, replacement in sorted_replacements:
        new_text += original_text[current_index:start] + replacement
        length_difference += len(replacement) - (end - start)
        current_index = end
        
    new_text += original_text[current_index:]
    return new_text, length_difference

original_text = "abcdefghi"
replacements = [(1, 4, "XYZ"), (5, 8, "12345")]
new_text, length_difference = replace_substrings(original_text, replacements)

with open("./datasets/ICTPE_v2/ICTPE_v2.json",encoding="utf-8") as f:
    procedures = json.load(f)
for p in procedures:
    op,st,cond,check = 0,0,0,0
    nodes = p["node_list"]
    text = p["text"] 
    replacements = []
    for i,step in enumerate(nodes):
        start = step["start"]
        end = step["end"]
        if step["type"]=="operate":
            replace = "执行"+step["type_id"]
        elif step["type"]=="status":
            replace = "到达"+step["type_id"]
        elif step["type"]=="condition":
            replace = "满足"+step["type_id"]
        elif step["type"]=="check":
            replace = "执行"+step["type_id"]
        replacements.append((start,end,replace))
    new_text,diff =replace_substrings(text,replacements)   
    p["text"] = new_text
with open("./datasets/ICTPE_v2/ICTPE_templated.json","w",encoding="utf-8") as f1:
    json.dump(procedures,f1,ensure_ascii=False)