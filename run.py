import json
from module.data import load_data
from module.embeddings import Emebeddings
from ipdb import set_trace as bp
from module.dimension_reduction._umap import UMAP
from loguru import logger

dataset_name = "allenai/c4"
if __name__ == "__main__":
    model_config = json.load(open("config/model.json"))
    # dataset_config = json.load(open("config/data.json"))
    file_path = 'train.json'
    dataset=[]
    with open(file_path, 'r', encoding='utf-8') as f:
            dataset=json.load(f)
    print(dataset[0])
    # dataset = json.load('en.noblocklist/c4-train.00000-of-01024.json')
    sentence_embeddings=[]
    for model in model_config:
        model_name = model["model_name"]
        embedding_dim = model["embedding_dim"]
        logger.info(f"processing {model_name} with dim {embedding_dim}")
        embedding_model = Emebeddings(model_name)
        # embedding_model.cuda()
        max_length = embedding_model.model.tokenizer.model_max_length
        # embeddings = model.encode(dataset)  
        # 截断每条文本
        texts = [text["text"][:max_length] for text in dataset if "text" in text]


        embeddings = embedding_model.encode(texts[:100])


        bp()
        # 使用 UMAP 进行降维
        umap = UMAP(n_components=2, random_state=42)  # 降维到二维
        embeddings_2d = umap.fit_transform(embeddings)
        
        # 可视化
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            embeddings_2d[:, 0], embeddings_2d[:, 1], cmap='viridis', alpha=0.7
        )
        plt.colorbar(scatter, label='Embedding Density')
        plt.title(f'UMAP Projection of {model_name} Embeddings')
        plt.xlabel('UMAP-1')
        plt.ylabel('UMAP-2')
        
        # 保存图片
        output_path = os.path.join(output_dir, f"umap_{model_name}_embeddings.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()  
        
        logger.info(f"Visualization for {model_name} saved at {output_path}")
    # # TODO 
    # embeddings_2d = umap.plot(sentence_embeddings,label=...)
        
