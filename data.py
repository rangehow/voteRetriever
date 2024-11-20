import json

# 替换为您的输入 JSON 文件路径
input_json_path = '/mnt/ljy/metb/voteRetriever/en.noblocklist/c4-train.00000-of-01024.json'

output_json_path = 'train.json'


data = []

# 逐行读取并解析 JSON 文件
with open(input_json_path, 'r', encoding='utf-8') as file:
    for line in file:
        try:
            json_obj = json.loads(line)
            # 提取每行 JSON 中的 text 字段
            if 'text' in json_obj:
                data.append({"text": json_obj['text']})
            else:
                print(f"Line missing 'text' field: {line.strip()}")
        except json.JSONDecodeError as e:
            print(f"Invalid JSON line: {e}")

# 将结果写入新的 JSON 文件
with open(output_json_path, 'w', encoding='utf-8') as json_file:
    json.dump(data, json_file, indent=4, ensure_ascii=False)

print(f"Successfully converted {input_json_path} to {output_json_path}")
