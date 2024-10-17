import pandas as pd
import json
from sklearn.model_selection import train_test_split

# Load the Excel data
df = pd.read_excel('G:/xjtu/llm/project/command/output/command_one_all_8_14.xlsx')

# Prepare the JSON structure
json_data = []

# Template for the instruction
instruction_template = "A level-one command is an instruction where the object is not clearly specified (it should not include the category of objects to be grabbed or the specific names of the objects to be grabbed) or the operation is unclear. （1）Extract all objects from the image.（2）Combine the image with universal world knowledge to describe the attributes and characteristics of the objects (must include information such as the object's purpose, color, location, classification, etc.).（3）Based on the image and object descriptions, along with universal knowledge, generate a level-one command. "

# Convert dataframe rows to JSON entries
for index, row in df.iterrows():
    print(index)
    output_text = row['command1']
    image_path = f"/root/autodl-tmp/data_new/rgb1/{row['image']}"

    entry = {
        "instruction": instruction_template,
        "input": "",
        "output": output_text,
        "images": [image_path]
    }

    json_data.append(entry)

# Save all data to a JSON file
with open('G:/xjtu/llm/project/command/output/实验/8-16/command1_all_data.json', 'w', encoding='utf-8') as f:
    json.dump(json_data, f, ensure_ascii=False, indent=4)

# Split data into training and testing sets
train_data, test_data = train_test_split(json_data, test_size=0.2, random_state=42)

# Save the training data to a JSON file
with open('G:/xjtu/llm/project/command/output/实验/8-16/command1_train_data.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)

# Save the testing data to a JSON file
with open('G:/xjtu/llm/project/command/output/实验/8-16/command1_test_data.json', 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=4)

print("JSON files created successfully!")
