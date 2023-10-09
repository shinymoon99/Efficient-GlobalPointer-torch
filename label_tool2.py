
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved. #复杂知识 - 流程知识标注
from collections import namedtuple
import streamlit as st
from utils.util import OrJson, OrJsonLine
from utils.util import get_all_span_start_end_list
import graphviz
import config


# 页面总配置
st.set_page_config(
    page_title="复杂知识-流程知识标注",
    layout="wide"
)
change_text = """
<style>
div[data-baseweb="notification"] {padding-top: 5px; padding-bottom:5px;} 
input[aria-label="页码"] {padding-top: 5px; padding-bottom:5px;}

.streamlit-expanderHeader {height: 0px; padding-bottom:16px; padding-top:0px;visibility:collapse;} 
div.row-widget.stRadio > div{flex-direction: row; justify-content: left;} 
div.row-widget.stRadio {flex-direction: row; justify-content: left;}
</style>
}
"""
st.markdown (change_text, unsafe_allow_html=True)
FLOW_TYPE = ["status", "operate", "check", "condition"]
FLOW_STR = {"status": "状态", "operate": "操作", "check": "判断", "condition": "条件"}
#@st.cache_resource
def get_flow_text_list():
    """获取所有的流程类文本"""
    all_knowledge = OrJsonLine.load(config.complex_knowledge_path)
    result = [item["text_after"] for item in all_knowledge if item["label"] == "流程知识"]
    return result

if "status3" not in st.session_state:
    st.session_state["status3"] = "当前第1个样本"
if "validate_status3" not in st.session_state: 
    st.session_state["validate_status3"] = ""
if "text_list3" not in st.session_state: 
    st.session_state["text_list3"] = get_flow_text_list()
if "text_index3" not in st.session_state: 
    st.session_state["text_index3"] = 0
if "text_span" not in st.session_state: st.session_state["text_span"] = ""
if "cur_start" not in st.session_state: st.session_state["cur_start"] = 0
if "cur_end" not in st.session_state: st.session_state["cur_end"] = 0

if "node_list" not in st.session_state:st.session_state["node_list"] = []
if "next_node_id" not in st.session_state: st.session_state["next_node_id"] = 0
if "edge_list" not in st.session_state: st.session_state["edge_list"] = []
if "cur_from_index" not in st.session_state: st.session_state["cur_from_index"] = 0
if "cur_to_index" not in st.session_state: st.session_state["cur_to_index"] = 1
## 当有多个text_span候选项的时候,进行切换选择 if "start_end_list" not in st.session_state: st.session_state["start_end_list"] = []
if "cur_text_input_index" not in st.session_state: st.session_state["cur_text_input_index"] = 0
if "cur_text_radio_index" not in st.session_state: st.session_state["cur_text_radio_index"] = 0



def clear_page_value():
    clear_status()
    clear_start_end()
    st.session_state["node_list"] = []
    st.session_state["edge_list"] = []
    st.session_state["next_node_id"] = 0
    st.session_state["cur_start"] = 0
    st.session_state["cur_end"] = 0
    st.session_state["text_span"] =""
    st.session_state["cur_from_index"]= 0
    st.session_state["cur_to_index"] =1

def previous_doc_click(): 
    """前一个问题"""
    clear_page_value()
    if st.session_state["text_index3"] > 0: 
        st.session_state["text_index3"] -= 1
        st.session_state["status3"] = f"""当前第{st.session_state["text_index3"] + 1} /
{len(st.session_state["text_list3"])}个样本"""
    else:
        st.session_state["status3"] =  "已经是第一个,无法向前了"


def jump_to(index):
    """跳转到index个问题"""
    clear_page_value()
    try:
        index= int(index)
        if not isinstance (index, int) or index < 1 or index > len(st.session_state["text_list3"]): 
            st.session_state["status3"] = f"""请输入一个[0, {len(st.session_state["text_list3"])}正整数"""
        else:

            st.session_state["text_index3"] = index - 1
            st.session_state['status3'] = f"""{st.session_state["text_index3"] + 1} /
{len(st.session_state["text_list3"])}个样本"""
    except:
        st.session_state["status3"] = "跳转出现错误"
def next_doc_click():
    """下一个问题"""
    clear_page_value()
    if st.session_state["text_index3"] < len(st.session_state["text_list3"]) - 1: 
        st.session_state["text_index3"] += 1
        st.session_state["status3"] = f"""当前第{st.session_state["text_index3"] + 1} /
{len(st.session_state["text_list3"])}个样本"""
    else:
        st.session_state["status3"] = "已经是最后一个了,无法向后了"




text_index = st.session_state["text_index3"]
cur_text = st.session_state["text_list3"][text_index]
col_left, col_status, col_jump, col_right = st.columns([2, 5, 3, 2])
with col_left:
    st.button("^", on_click=previous_doc_click)
with col_status:
    st.info(f"""{st.session_state["status3"]}""")
with col_jump:
    col1, col2 = st.columns([1, 1])
    with col1:
        index = st.text_input(label="", placeholder="", label_visibility="collapsed", key="page_number") 
    with col2:
        st.button(label="", on_click=jump_to, kwargs={"index": index})
with col_right:
    st.button("^", on_click=next_doc_click)
def select_cur_text():
    """从多个不同得候选文本中选择一个文本"""
    index = st.session_state["cur_text_radio"] 
    start_end_list = st.session_state["start_end_list"]
    st.session_state["cur_text_radio_index"] = index 
    st.session_state["cur_start"] = start_end_list[index][0] 
    st.session_state["cur_end"] = start_end_list[index][1] 
    start, end = start_end_list[index]
    node_index = st.session_state["cur_text_input_index"] 
    st.session_state["node_list"] [node_index].update({
"start": start,
"end": end
})
    st.session_state["cur_start"] = start 
    st.session_state["cur_end"] = end

last_start = 0
text_str = ""
for index, (start, end) in enumerate(st.session_state["start_end_list"]): 
    if index == st.session_state["cur_text_radio_index"]:
        text_str += "{cur_text[last_start: start]}:red[ [{cur_text[start:end]}]]"
    else:
        text_str += "{cur_text[last_start:start]}:red[{cur_text[start:end]}]" 
    last_start = end
text_str += cur_text[last_start:]
st.write(text_str)
if len(st.session_state["start_end_list"]) > 1:
    st.radio(key="cur_text_radio",label="存在多个文本，请选择其中一个",options=[i for i in range(len(st.session_state["start_end_list"]))], on_change=select_cur_text)
def clear_status():
    st.session_state["validate_status3"]=""
def clear_start_end():
    st.session_state["start_end_list"] = []
    st.session_state["cur_text_radio_index"] = 0
    st.session_state["cur_text_input_index"] = 0
    st.session_state["cur_text_radio"] = 0


def show_text_span(flow_index, text_input_key):
    """对文本先后进行渲染"""
    clear_status()
    clear_start_end()
    cur_text_radio_index = 0
    text_span= st.session_state[text_input_key]
    start_end_list = get_all_span_start_end_list(text_span, cur_text) 
    st.session_state["start_end_list"] = start_end_list
    if start_end_list:
        start, end = start_end_list[cur_text_radio_index] 
        st.session_state["node_list"] [flow_index].update({"text_span": text_span,
        "start": start,
        "end": end
        })
        st.session_state["cur_start"] = start 
        st.session_state["cur_end"] = end
        st.session_state["cur_text_input_index"] = flow_index
    else:


        st.session_state["node_list"][flow_index].update({
                    "text_span": "",
                "start": 0, "end": 0
        })

        st.session_state["cur_start"] = 0
        st.session_state["cur_end"] = 0
        st.session_state["cur_text_input_index"] = flow_index


def add_node(flow_type): 
    """向流程中增加不同类型的步骤"""
    clear_status()
    clear_start_end()
    if validate_node():
        id_str = f"""{FLOW_STR[flow_type]}_{st.session_state["next_node_id"]}"""
        st.session_state["node_list"].append({
            "id": id_str,
            "type": flow_type, 
            "text_span": "",
            "start": 0,
            "end": 0
    })

        st.session_state["next_node_id"] += 1
def del_node(node_id):
    """从流程中删除某个步骤"""
    clear_status()
    for index, edge_info in enumerate(st.session_state["edge_list"]):
        if node_id == edge_info["from"] or node_id == edge_info["to"]:
            del st.session_state["edge_list"][index]
    for index, node_info in enumerate(st.session_state["node_list"]): 
        if node_info["id"] == node_id:
            del st.session_state["node_list"][index]
    if len(st.session_state["node_list"]) == 0:
        st.session_state["next_node_id"] = 0

def add_edge():
    """添加各个不同的流程之间的依赖"""
    clear_status()
    if validate_node():
        for index, edge_info in enumerate(st.session_state["edge_list"]):
            if {
"from": st.session_state["edge_from"],
"to": st.session_state["edge_to"]
} == edge_info:
                st.session_state[
"validate_status3"] = f"""{st.session_state["edge_from"]}, {st.session_state["edge_to"]} 已经存在"""
                return
        st.session_state["edge_list"].append({"from": st.session_state["edge_from"],"to": st.session_state["edge_to"]
})
        node_list = get_flow_node_list()
        st.session_state["cur_from_index"] = node_list.index(st.session_state["edge_to"])
        if st.session_state["cur_from_index"] < len(node_list) - 1:
            st.session_state["cur_to_index"] = st.session_state["cur_from_index"] + 1


def del_edge():
    """删除各个不同流程之间的依赖"""
    clear_status()
    for index, edge_info in enumerate(st.session_state["edge_list"]):
        if {
"from": st.session_state["edge_from"],
"to": st.session_state["edge_to"]
} == edge_info:
            del st.session_state["edge_list"][index]
            return
st.session_state["validate_status3"] = f"""{st.session_state["edge_from"]}, {st.session_state["edge_to"]}
连接边不存在"""
def get_flow_node_list():
    """获取当前标注中所有节点"""
    result_list = list()
    for index, node_info in enumerate (st.session_state["node_list"]):
        result_list.append(node_info["id"])
    return result_list


def get_flow_edge_list(): 
    """获取步骤之间的关系"""
    result_list = list()
    for index, edge_info in enumerate(st.session_state["edge_list"]): result_list.append([edge_info["from"], edge_info["to"]])
    return result_list

def validate_node():
    for index, node_info in enumerate (st.session_state["node_list"]): 
        if node_info["text_span"] == "":
            st.session_state["validate_status3"]=f"请先输入{node_info['id']}的文本"
            return False
    return True

def validate_edge():
    linked_node_set = set()
    for edge_info in st.session_state["edge_list"]:
        linked_node_set.add(edge_info["from"]) 
        linked_node_set.add(edge_info["to"])
    all_node_set = set([node_info["id"] for node_info in st.session_state["node_list"]])
    isolation_node_set = all_node_set - linked_node_set
    if len(isolation_node_set) > 0:
        st.session_state["validate_status3"] =f"以下节点还未连接,{isolation_node_set}"
        return False

def get_sample_json_obj():
    return {
    "text": cur_text,
"node_list": st.session_state["node_list"], "edge_list": get_flow_edge_list()
}

def save_sample():
    if validate_edge() and validate_node():
        OrJsonLine.dump([get_sample_json_obj()], "./data/complex_data/complex_data_label_flow.json", "a")
        next_doc_click()


col1, col2, col3, col4, col6, col7 = st.columns([1, 1, 1, 1, 1, 1]) 
with col1:
    st.button("添加状态", on_click=add_node, kwargs={"flow_type": FLOW_TYPE[0]}, use_container_width=True)
with col2:
    st.button("添加操作", on_click=add_node, kwargs={"flow_type": FLOW_TYPE[1]}, use_container_width=True)
with col3:
    st.button("添加判断", on_click=add_node, kwargs={"flow_type": FLOW_TYPE [2]}, use_container_width=True)
with col4:
    st.button("添加条件", on_click=add_node, kwargs={"flow_type": FLOW_TYPE[3]}, use_container_width=True)
with col6:
    st.button("重置", on_click=clear_page_value, use_container_width=True)
with col7:
    st.button("保存", on_click=save_sample, use_container_width=True)



if len(st.session_state["node_list"]) > 0:
    with st.expander (label="", expanded=True):
        for index, flow_info in enumerate(st.session_state["node_list"]):
            col1, col2, col3, col4 = st.columns([1, 2, 1, 1])
    with col1:
        st.button(flow_info["id"], disabled=True, use_container_width=True)
    with col2:
        st.text_input(label=f"text_span_{index}", key=f"text_span_{index}",
label_visibility="collapsed",value=flow_info["text_span"],
on_change=show_text_span,
kwargs={"text_input_key": f"text_span_{index}", "flow_index": index})
    with col3:
        start = flow_info["start"]
        end = flow_info["end"]
        st.text_input(label=f"text_span_index_{index}", key=f" text_span_index_{index}",
label_visibility="collapsed",
value=f" {start}, {end}")
    with col4:
        st.button(label="B", key=f"del_text_span_{index}", on_click=del_node, kwargs={"node_id": flow_info["id"]}, use_container_width=True)
if len(st.session_state["node_list"]) > 1:
    with st.expander (label="", expanded=True):
        col1, col2 = st.columns([1, 4])
    with col1:
        st.selectbox (label=f"edge_from", key="edge_from", label_visibility="collapsed",
options=get_flow_node_list(),
index=st.session_state["cur_from_index"])
        st.selectbox(label=f"edge_to", key="edge_to", label_visibility="collapsed",
options=get_flow_node_list(),
index=st.session_state["cur_to_index"])
    with col2:
        edge_list = get_flow_edge_list()
        st.text_area(label="edge_list", value=edge_list, label_visibility="collapsed") 
        col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        st.button(label="", on_click=add_edge, use_container_width=True)
    with col2:
        st.button(label="", on_click=del_edge, use_container_width=True)
if st.session_state["validate_status3"] != "":
    st.error(st.session_state["validate_status3"])
with st.expander (label="", expanded=True):
    graph = graphviz.Digraph()
    graph.attr(rankdir="LR")
    # 渲染节点
    for index, flow_info in enumerate(st.session_state["node_list"]): 
        if flow_info["type"]== FLOW_TYPE[0]:
            graph.node (flow_info["id"], shape="box")
        if flow_info["type"] == FLOW_TYPE[1]:
            graph.node(flow_info["id"], shape="box")
        if flow_info["type"] == FLOW_TYPE[2]:
            graph.node(flow_info["id"], shape="diamond")
        if flow_info["type"] == FLOW_TYPE[3]:
            graph.node(flow_info["id"], shape="box", fillcolr='#40e0d0')
    # 渲染链接
for index, edge_info in enumerate(st.session_state["edge_list"]): 
    graph.edge (edge_info["from"], edge_info["to"])
st.graphviz_chart(graph)
st.json(get_sample_json_obj())