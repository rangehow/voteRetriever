import json
import torch
from transformers import AutoModel
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# 禁用HF Hub Transfer
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "false"

# 加载模型
model = AutoModel.from_pretrained("/mnt/ljy/metb/NV-Embed-v2", trust_remote_code=True)
model.max_seq_length = 32768
model.tokenizer.padding_side = "right"

def add_eos(input_examples):
    """为输入添加结束标记"""
    return [input_example + model.tokenizer.eos_token for input_example in input_examples]

def load_sentences(file_path):
    """从JSON文件加载单个句子列表"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def compute_embeddings(texts, batch_size=32):
    """计算文本嵌入向量"""
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Computing Embeddings"):
        batch_texts = texts[i:i + batch_size]
        encoded_batch = model.encode(add_eos(batch_texts), batch_size=len(batch_texts), normalize_embeddings=True)
        embeddings.append(encoded_batch)
    return torch.cat(embeddings, dim=0)

def visualize_embeddings(embeddings, perplexity=30, save_path=None):
    """降维并绘制散点图"""
    reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    reduced_embeddings = reducer.fit_transform(embeddings.cpu().detach().numpy())

    plt.figure(figsize=(10, 7))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.5)
    plt.title("Sentence Embeddings Visualization")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    
    if save_path:
        plt.savefig(save_path, format='png', dpi=300)
        print(f"图像已保存到: {save_path}")
    else:
        plt.show()

def main():
    # 抽取后的文件路径
    file_path = '/mnt/ljy/metb/wikipedia/random_3000_samples.json'  # 包含单个句子的JSON文件
    sentences = load_sentences(file_path)
    
    # 计算嵌入
    embeddings = compute_embeddings(sentences)
    
    # 可视化嵌入
    visualize_embeddings(embeddings, perplexity=30, save_path="single_sentence_visualization.png")

if __name__ == "__main__":
    main()
