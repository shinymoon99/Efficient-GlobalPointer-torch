import os
from pathlib import Path
import json
def read_json_by_line(filepath):
    """
    Reads a JSON file line by line and returns a list of JSON objects.

    Args:
    filepath (str): The path to the JSON file.

    Returns:
    list: A list of JSON objects, one for each line in the file.
    """
    json_objects = []

    with open(filepath, 'r') as file:
        for line in file:
            try:
                # Parse the line as a JSON object
                json_obj = json.loads(line)
                json_objects.append(json_obj)
            except json.JSONDecodeError as e:
                # Handle JSON decoding errors if needed
                print(f"Error decoding JSON on line: {line.strip()}")
    
    return json_objects
def write_json_by_line(filepath, data_list):
    """
    Write JSON data line by line to a file.

    Args:
        filepath (str): The path to the file where JSON data will be written.
        data_list (list): A list of dictionaries, where each dictionary represents JSON data.

    Returns:
        bool: True if the operation was successful, False otherwise.
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as file:
            for data in data_list:
                # Serialize the dictionary to JSON and write it to the file followed by a newline.
                json.dump(data, file, ensure_ascii=False)
                file.write('\n')
        return True
    except Exception as e:
        print(f"Error writing JSON data to {filepath}: {str(e)}")
        return False
def append_json_by_line(filepath, data_list):
    """
    Write JSON data line by line to a file.

    Args:
        filepath (str): The path to the file where JSON data will be written.
        data_list (list): A list of dictionaries, where each dictionary represents JSON data.

    Returns:
        bool: True if the operation was successful, False otherwise.
    """
    try:
        with open(filepath, 'a', encoding='utf-8') as file:
            for data in data_list:
                # Serialize the dictionary to JSON and write it to the file followed by a newline.
                json.dump(data, file, ensure_ascii=False)
                file.write('\n')
        return True
    except Exception as e:
        print(f"Error writing JSON data to {filepath}: {str(e)}")
        return False
def get_all_span_start_end_list(text_span, text): 
    """获取在文本中出现的所有片段的起始位置""" 
    text_span= text_span.strip()
    if text_span == "": 
        return []
    start_end_list = list()
    index = 0
    while True:
        start = text.find(text_span, index)
        if start != -1:
            start_end_list.append([start, start+len(text_span)]) 
            index = start + len(text_span)
        else:
            break
    return start_end_list
class IO:
    """IO的基类"""
    @staticmethod
    def is_valid_file(filepath):
        filepath = Path(filepath)
        return filepath.exists() and filepath.stat().st_size>0
    def load(path):
        raise NotImplementedError
    def dump(data,path):
        raise NotImplementedError
class OrJson(IO):
    """使用orison实现的json格式数据的存储与加载"""
    @staticmethod
    def load(path):
        with open(path,"rb")as rf:
            data = orjson.Loads(rf.read())
        return data
    @staticmethod
    def loads(jsonLine):
        return orjson.Loads(jsonLine)
    @staticmethod
    def dump(data, path):
        with open(path,"w", encoding='utf8')as wf:
            wf.write(orjson.dumps(data, option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS).decode())
    @staticmethod
    def dumps(data):
        return orjson.dumps(data, option=orjson.OPT_NON_STR_KEYS).decode()

class OrJsonLine(IO):
    @staticmethod
    def load(path):
        with open(path, encoding="utf-8") as rf: 
            lines = rf.read().splitlines() 
        return [orjson.loads(l) for l in lines]
    @staticmethod
    def dump(instances, path, mode="w"):
        assert type (instances) == list
        lines = [orjson.dumps (d, option=orjson.OPT_NON_STR_KEYS).decode() for d in instances] 
        with open(path, mode, encoding="utf-8") as wf:
            wf.write('\n'.join(lines) + "\n")