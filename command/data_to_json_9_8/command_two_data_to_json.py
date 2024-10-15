import pandas as pd
import json
from sklearn.model_selection import train_test_split

# Load the Excel data
df = pd.read_excel('G:/xjtu/llm/project/command/output/output_8_14_command2_generate.xlsx')

# Prepare the JSON structure
json_data = []

# Template for the instruction
instruction_template = "A Level-two instruction should include a general object category (which must correspond to more than one object) and a clear single step of operation. (1)First, extract all objects from the image.(2)Use universal world knowledge to provide a detailed description of the objects' features (must include characteristics such as length, width, height, color, shape, etc.).(3)Classify the objects using universal world knowledge.(4)Based on the classification results and object descriptions, generate an appropriate level-two command."
# Convert dataframe rows to JSON entries
for index, row in df.iterrows():
    print(index)
    output_text = row['command2']
    image_path = f"/root/autodl-tmp/data_new/rgb1/{row['image']}"

    entry = {
        "instruction": instruction_template,
        "input": "",
        "output": output_text,
        "images": [image_path]
    }

    json_data.append(entry)

# Save all data to a JSON file
with open('G:/xjtu/llm/project/command/output/实验/8-16/command2_all_data.json', 'w', encoding='utf-8') as f:
    json.dump(json_data, f, ensure_ascii=False, indent=4)

# Split data into training and testing sets
train_data, test_data = train_test_split(json_data, test_size=0.2, random_state=42)

# Save the training data to a JSON file
with open('G:/xjtu/llm/project/command/output/实验/8-16/command2_train_data.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)

# Save the testing data to a JSON file
with open('G:/xjtu/llm/project/command/output/实验/8-16/command2_test_data.json', 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=4)

print("JSON files created successfully!")
