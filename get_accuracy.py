import json
def calculate_node_accuracy(model_nodes, gold_nodes):
    # Convert the node lists to sets for easier comparison
    model_node_set = set((node["type_id"], node["text_span"]) for node in model_nodes)
    gold_node_set = set((node["type_id"], node["text_span"]) for node in gold_nodes)

    # Calculate accuracy as the Jaccard similarity between model and gold nodes
    common_nodes = model_node_set.intersection(gold_node_set)
    accuracy = len(common_nodes) / (len(model_node_set) + len(gold_node_set) - len(common_nodes))

    return accuracy

def calculate_edge_accuracy(model_edges, gold_edges):
    # Convert the edge lists to sets for easier comparison
    model_edge_set = set(tuple(edge) for edge in model_edges)
    gold_edge_set = set(tuple(edge) for edge in gold_edges)

    # Calculate accuracy as the Jaccard similarity between model and gold edges
    common_edges = model_edge_set.intersection(gold_edge_set)
    accuracy = len(common_edges) / (len(model_edge_set) + len(gold_edge_set) - len(common_edges))

    return accuracy

# List of data points and corresponding gold data
with open("./datasets/ICTPE_v2/ICTPE_dev.json","r",encoding="utf-8") as f:
    data_list =json.load(f)
with open("./outputs/ICTPE_test.json","r",encoding="utf-8") as f:
    gdata_list =json.load(f)

# Calculate overall node and edge accuracy
total_node_accuracy = 0.0
total_edge_accuracy = 0.0

for i,data_point in enumerate(data_list):
    model_data = data_point
    gold_data = gdata_list[i]  # Assuming gold data is provided in the same format as model data
    
    node_accuracy = calculate_node_accuracy(model_data["node_list"], gold_data["node_list"])
    edge_accuracy = calculate_edge_accuracy(model_data["edge_list"], gold_data["edge_list"])
    
    total_node_accuracy += node_accuracy
    total_edge_accuracy += edge_accuracy

# Calculate the average accuracy over all data points
average_node_accuracy = total_node_accuracy / len(data_list)
average_edge_accuracy = total_edge_accuracy / len(data_list)

print(f"Overall Node Accuracy: {average_node_accuracy * 100:.2f}%")
print(f"Overall Edge Accuracy: {average_edge_accuracy * 100:.2f}%")
