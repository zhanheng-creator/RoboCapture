import pandas as pd
import json
from sklearn.model_selection import train_test_split

# Load the Excel data
df = pd.read_excel('G:/xjtu/llm/project/command/output/output_8_14_command3_generate.xlsx')

# Prepare the JSON structure
json_data = []

# Template for the instruction

instruction_template = "Please generate a Level-four instruction based on the task instruction and the picture scene. A Level-four instruction is a sequence of actions that can complete a task. The actions only include 'grab' and 'place.' The action sequence template is as follows: <(action1, obj1), (action2, obj2, Direction)>. The Direction can be front, back, left, right, or above.Just output the Level-four instruction without explanation or any additional content."
# Convert dataframe rows to JSON entries
for index, row in df.iterrows():
    print(index)
    output_text = row['assistant_disassemble_res']
    input_command = row['assistant_ins_res']
    image_path = f"/root/autodl-tmp/data_new/rgb1/{row['image']}"

    entry = {
        "instruction": instruction_template,
        "input": input_command,
        "output": output_text,
        "images": [image_path]
    }

    json_data.append(entry)

# Save all data to a JSON file
with open('G:/xjtu/llm/project/command/output/实验/9-8无多步提示词的test/command4_all_data.json', 'w', encoding='utf-8') as f:
    json.dump(json_data, f, ensure_ascii=False, indent=4)

# Split data into training and testing sets
train_data, test_data = train_test_split(json_data, test_size=0.2, random_state=42)

# Save the training data to a JSON file
with open('G:/xjtu/llm/project/command/output/实验/9-8无多步提示词的test/command4_train_data.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)

# Save the testing data to a JSON file
with open('G:/xjtu/llm/project/command/output/实验/9-8无多步提示词的test/command4_test_data.json', 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=4)

print("JSON files created successfully!")
