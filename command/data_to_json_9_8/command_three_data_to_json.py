import pandas as pd
import json
from sklearn.model_selection import train_test_split

# Load the Excel data
df = pd.read_excel('G:/xjtu/llm/project/command/output/output_8_14_command3_generate.xlsx')

# Prepare the JSON structure
json_data = []

# Template for the instruction

instruction_template = "A Level-three instruction is an instruction to perform a specific operation on an object with a clear name, and it must include a specific placement position (e.g., left, right, front, back, or on top of another specific object), such as 'place the apple on the book.' (1)Extract all objects from the image.(2)Provide a detailed description of the objects on the table based on the image. The description must include the length, width, height, shape, and thickness of the objects, as well as whether the objects are placed horizontally or vertically, incorporating both the image and common sense in the description.(3)Based on the image and description, identify all objects suitable for grasping and placing by a two-finger gripper. (Suitable grasping characteristics: the thickness of the object on the table must be greater than 3 cm, and the maximum width in the grasping direction must be less than 8 cm. The object to be placed must be suitable for placement, and the object to be grasped should not already be located on top of the object to be placed. Thickness comparison example: 2 cm is greater than 0.3 cm, 2.5 cm is greater than 2.15 cm. The thickness of the object suitable for grasping must be greater than 3 cm.)(4)Based on the description, image, and object list, extract pairs of objects suitable for operation by the two-finger gripper in the image [object1, object2]. Object1 is the object to be grasped, and object2 is the location where object1 is to be placed. As a reference example: object2 can be a tray (corresponding to placing object1 into the tray), a book (corresponding to placing object1 on the book), an apple (corresponding to placing object1 to the right of the apple), or a wooden block (corresponding to stacking object1 on the wooden block).(5)Choose an appropriate task for the pair of objects in the image and generate a level-three command suitable for operation by the two-finger gripper."
# Convert dataframe rows to JSON entries
for index, row in df.iterrows():
    print(index)
    output_text = row['assistant_ins_res']
    image_path = f"/root/autodl-tmp/data_new/rgb1/{row['image']}"

    entry = {
        "instruction": instruction_template,
        "input": "",
        "output": output_text,
        "images": [image_path]
    }

    json_data.append(entry)

# Save all data to a JSON file
with open('G:/xjtu/llm/project/command/output/实验/8-16/command3_all_data.json', 'w', encoding='utf-8') as f:
    json.dump(json_data, f, ensure_ascii=False, indent=4)

# Split data into training and testing sets
train_data, test_data = train_test_split(json_data, test_size=0.2, random_state=42)

# Save the training data to a JSON file
with open('G:/xjtu/llm/project/command/output/实验/8-16/command3_train_data.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)

# Save the testing data to a JSON file
with open('G:/xjtu/llm/project/command/output/实验/8-16/command3_test_data.json', 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=4)

print("JSON files created successfully!")
