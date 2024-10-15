import pandas as pd
import json
from sklearn.model_selection import train_test_split

# Load the Excel data
df = pd.read_excel('G:/xjtu/llm/project/command/output/output_8_15_command2_dismantle.xlsx')

# Prepare the JSON structure
json_data = []

# Template for the instruction
# instruction_template = "A Level-two instruction should include a general object category (which must correspond to more than one object) and a clear single step of operation.A Level-three instruction is an instruction that involves a specific operation on an object with a clearly defined name, and the Level-three instruction must have a clearly defined placement location (in relation to another specific object, e.g., left, right, front, back, above). For example, 'Place the apple on top of the book.' Please break down the Level-two instruction for this image into Level-three instructions. 1. Extract the generalized objects from the level-two command.2. Based on the image, identify all specific objects that fall under the generalized object mentioned in the level-two command.3. Using universal world knowledge, provide a detailed description of the specific objects' features (including length, width, height, shape, color, etc.).4. Based on the level-two command, generate a specific level-three command for each specific object.Please break down the command according to the items present in the image.Just output the decomposed commands without explanation or any additional content.The Level-two instruction for this image is"
#二级指令应包括一个通用的物体类别（该类别必须对应多个物体）和一个明确的单步操作。三级指令是指涉及对具有明确名称的物体进行具体操作的指令，且三级指令必须具有明确的放置位置（相对于另一具体物体的位置，例如左、右、前、后、上方）。例如，‘将苹果放在书的上方。’ 根据图片，将对应的二级指令分解为三级指令。该图像的二级指令是：
instruction_template="A Level-two instruction should include a general object category (which must correspond to more than one object) and a clear single step of operation.A Level-three instruction is an instruction that involves a specific operation on an object with a clearly defined name, and the Level-three instruction must have a clearly defined placement location (in relation to another specific object, e.g., left, right, front, back, above). For example, 'Place the apple on top of the book.'Based on the image, break down the corresponding Level-two instruction into Level-three instructions. The Level-two instruction for this image is:"
# Convert dataframe rows to JSON entries
for index, row in df.iterrows():
    print(index)
    command2 = row['command2']
    command3 = row['command3']
    image_path = f"/root/autodl-tmp/data_new/rgb1/{row['image']}"

    entry = {
        "instruction": instruction_template,
        "input": command2,
        "output": command3,
        "images": [image_path]
    }

    json_data.append(entry)

# Save all data to a JSON file
with open('G:/xjtu/llm/project/command/output/实验/9-8无多步提示词的test/command2_dismantle_all_data.json', 'w', encoding='utf-8') as f:
    json.dump(json_data, f, ensure_ascii=False, indent=4)

# Split data into training and testing sets
train_data, test_data = train_test_split(json_data, test_size=0.2, random_state=42)

# Save the training data to a JSON file
with open('G:/xjtu/llm/project/command/output/实验/9-8无多步提示词的test/command2_dismantle_train_data.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)

# Save the testing data to a JSON file
with open('G:/xjtu/llm/project/command/output/实验/9-8无多步提示词的test/command2_dismantle_test_data.json', 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=4)

print("JSON files created successfully!")
