import json

# 定义n和k的值
n = 175  # 列表中元素的数量
k = 35  # 每个元素的值

# 创建列表
episode_length = [k for _ in range(n)]

# 将列表转换为字典，以便于保存为JSON格式
data = {
    "episode_length": episode_length
}

# 写入到文件
with open('dataset_info.json', 'w') as file:
    json.dump(data, file, indent=4)

print("文件已成功生成！")