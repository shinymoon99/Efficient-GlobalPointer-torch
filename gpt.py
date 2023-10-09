import openai
import json
import codecs
import time
openai.api_key = "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"
openai.api_base = "https://quiet-rook-90.deno.dev/v1"

def readInput(data):
    inputs = []
    for i in range(len(data)):
        inputs.append(data[i]["text"])
    return inputs

## read prompt and form input to chatgpt to generate result
with open("./prompt_fcode.txt","r",encoding="utf-8") as f:
    t= f.read()
# get a str list
with open("./datasets/ICTPE_v2/ICTPE_dev.json","r",encoding="utf-8") as f1:
    data = json.load(f1)
inputs = readInput(data)
outputs = ""
#outputs = "[\n"
print("input_num:"+str(len(inputs)))
print()
for i in range(len(inputs)):
    start_time = time.time()
    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": t+"\n输入:"+inputs[i]}
    ]
    )
    api_response = completion.choices[0].message
    api_response_content =api_response["content"]
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(str(i)+" input complete ")
    # Print the elapsed time
    print("Elapsed time: {:.2f} seconds".format(elapsed_time))
    print(api_response_content)
    outputs+=(api_response_content+",\n")
    # Add a sleep interval between questions
    if i < len(inputs) - 1:
        if elapsed_time<20:
            time.sleep(21-elapsed_time)  # Sleep for at least 20 seconds
#outputs=outputs[:-2]+"]"
    #TODO： [] added needed for output
with open("./output/fcode_out.txt","w",encoding="utf-8") as f2:
    f2.write(outputs)
