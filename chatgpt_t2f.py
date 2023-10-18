import openai
import json
import codecs
import time
import signal
import sys
#from eval.eval import getAccuracy,getALLNodeList,getEdgeList
openai.api_key = ""
openai.api_base = "https://quiet-rook-90.deno.dev/v1"
def append_list_to_json_by_line(file_path, data_list):
    """
    Write a list of elements to a JSON file, with each element on a separate line.
    
    Args:
        file_path (str): The path to the JSON file.
        data_list (list): The list of data elements to write to the file.
    """
    with open(file_path, 'a') as file:
        for item in data_list:
            json.dump(item, file)
            file.write('\n')
def readInput(data):
    inputs = []
    for i in range(len(data)):
        inputs.append(data[i]["text"])
    return inputs
def save_results_and_exit(signum, frame):
    # Signal handler to save results and exit gracefully
    print("Received interrupt signal. Saving results and exiting...")
    append_list_to_json_by_line("./outputs/t2f.json", response_all)
    sys.exit(1)
## read prompt and form input to chatgpt to generate result
with open("./prompts/prompt_fcode.txt","r",encoding="utf-8") as f:
    t= f.read()
# get a str list
with open("./datasets/ICTPE_v2/ICTPE_v2.json","r",encoding="utf-8") as f1:
    data = json.load(f1)
inputs = readInput(data)
outputs = ""
response_all = []
#outputs = "[\n"
print("input_num:"+str(len(inputs)))

# Register the signal handler for Ctrl+C (SIGINT)
signal.signal(signal.SIGINT, save_results_and_exit)

for i in range(len(inputs)):
    start_time = time.time()
    
    while True:
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": t+"\n输入:"+inputs[i]+"\n输出:(请按照python格式输出)"}
                ]
            )

            api_response = completion.choices[0].message
            api_response_content = api_response["content"]
            response = {"id": i, "response": api_response_content}
            response_all.append(response)

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(str(i) + " input complete ")
            # Print the elapsed time
            print("Elapsed time: {:.2f} seconds".format(elapsed_time))
            print(api_response_content)
            break  # Exit the retry loop when the call is successful
        except Exception as e:
            print(f"An exception occurred for input {i}: {str(e)}")
            time.sleep(5)  # Wait for a few seconds before retrying

    # Add a sleep interval between questions
    if i < len(inputs) - 1:
        if elapsed_time < 20:
            time.sleep(21 - elapsed_time)  # Sleep for at least 20 seconds

# Append the responses to the JSON file after the loop
append_list_to_json_by_line("./outputs/t2f.json", response_all)