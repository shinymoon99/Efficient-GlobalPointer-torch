import tkinter as tk
import json

class JSONSelector:
    def __init__(self, root, json_file):
        self.root = root
        self.json_file = json_file
        self.data = []
        self.selected_data = []
        self.unselected_data = []
        self.load_data()
        self.index_box = tk.Text(root, wrap=tk.WORD, width=5, height=1)
        self.index_box.pack()
        self.text_box = tk.Text(root, wrap=tk.WORD, width=250, height=50)
        self.text_box.pack()
        
        self.yes_button = tk.Button(root, text="Yes", command=self.select_data)
        self.yes_button.pack()
        
        self.no_button = tk.Button(root, text="No", command=self.skip_data)
        self.no_button.pack()
        
        self.view_button = tk.Button(root, text="View Selected", command=self.view_selected)
        self.view_button.pack()
        
        self.save_button = tk.Button(root, text="Save Selected", command=self.save_selected)
        self.save_button.pack()
        
        self.current_index = 0
        self.show_data()

    def load_data(self):
        try:
            with open(self.json_file, 'r') as file:
                self.data = json.load(file)
        except FileNotFoundError:
            print("JSON file not found!")

    def show_data(self):
        if self.current_index < len(self.data):
            item = self.data[self.current_index]
            formatted_data = json.dumps(item, indent=4,ensure_ascii=False)
            self.text_box.delete(1.0, tk.END)
            self.text_box.insert(tk.END, formatted_data)
            self.index_box.delete(1.0, tk.END)
            self.index_box.insert(tk.END, str(self.current_index)+"/"+str(len(self.data)))
        else:
            self.text_box.delete(1.0, tk.END)
            self.text_box.insert(tk.END, "No more data")

    def select_data(self):
        if self.current_index < len(self.data):
            self.selected_data.append(self.data[self.current_index])
            self.current_index += 1
            self.show_data()

    def skip_data(self):
        if self.current_index < len(self.data):
            self.unselected_data.append(self.data[self.current_index])            
            self.current_index += 1
            self.show_data()

    def view_selected(self):
        selected_window = tk.Toplevel(self.root)
        selected_window.title("Selected Data")
        
        selected_text = tk.Text(selected_window, wrap=tk.WORD, width=50, height=10)
        selected_text.pack()
        
        for item in self.selected_data:
            formatted_data = json.dumps(item, indent=4,ensure_ascii=False)
            selected_text.insert(tk.END, formatted_data + "\n")

    def save_selected(self):
        with open("selected.json", "w") as file:
            json.dump(self.selected_data, file, indent=4,ensure_ascii=False)
        with open("unselected.json", "w") as file:
            json.dump(self.unselected_data, file, indent=4,ensure_ascii=False)
        print("Selected data saved to 'selected.json'")

if __name__ == "__main__":
    root = tk.Tk()
    root.title("JSON Data Selector")

    json_file = "data.json"  # 你的JSON文件路径

    app = JSONSelector(root, json_file)

    root.mainloop()
