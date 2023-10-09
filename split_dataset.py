import json
import random
def split_json_dataset(json_file_path, train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1, random_seed=None):
    with open(json_file_path, 'r',encoding="utf-8") as f:
        data = json.load(f)

    if random_seed is not None:
        random.seed(random_seed)
        random.shuffle(data)

    total_samples = len(data)
    train_samples = int(total_samples * train_ratio)
    dev_samples = int(total_samples * dev_ratio)
    test_samples = total_samples - train_samples - dev_samples

    train_data = data[:train_samples]
    dev_data = data[train_samples:train_samples + dev_samples]
    test_data = data[train_samples + dev_samples:]

    return train_data, dev_data, test_data

# Usage
json_file_path = './datasets/ICTES_v2/ICTES.json'
train_data, dev_data, test_data = split_json_dataset(json_file_path)

#Now you have your data split into train, dev, and test datasets
with open("./datasets/ICTES_v2/ICTES_train.json","w",encoding="utf-8") as f1:
    json.dump(train_data,f1,ensure_ascii=False)
with open("./datasets/ICTES_v2/ICTES_dev.json","w",encoding="utf-8") as f2:
    json.dump(dev_data,f2,ensure_ascii=False)
with open("./datasets/ICTES_v2/ICTES_test.json","w",encoding="utf-8") as f3:
    json.dump(test_data,f3,ensure_ascii=False)


# with open("./datasets/ICT_train.json","w",encoding="utf-8") as f1:
#     for item in train_data:
#         json.dump(item, f1,ensure_ascii=False)  # Write JSON object to the file
#         f1.write('\n')       # Add a newline to separate objects
# with open("./datasets/ICT_dev.json","w",encoding="utf-8") as f2:
#     for item in dev_data:
#         json.dump(item, f2,ensure_ascii=False)  # Write JSON object to the file
#         f2.write('\n')       # Add a newline to separate objects
# with open("./datasets/ICT_test.json","w",encoding="utf-8") as f3:
#     for item in test_data:
#         json.dump(item, f3,ensure_ascii=False)  # Write JSON object to the file
#         f3.write('\n')       # Add a newline to separate objects