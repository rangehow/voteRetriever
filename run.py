import json
from module.data import load_data
from module.embeddings import Emebeddings
from ipdb import set_trace as bp
from module.dimension_reduction._umap import UMAP
from loguru import logger
import numpy as np

dataset_name = "allenai/c4"
if __name__ == "__main__":
    model_config = json.load(open("config/model.json"))
    dataset_config = json.load(open("config/data.json"))
    print(dataset_config[dataset_name]["path"])
    # print(**dataset_config[dataset_name])
    dataset = load_data(**dataset_config[dataset_name])
    # dataset = json.load('en.noblocklist/c4-train.00000-of-01024.json')
    sentence_embeddings=[]
    for model in model_config:
        model_name = model["model_name"]
        embedding_dim = model["embedding_dim"]
        logger.info(f"processing {model_name} with dim {embedding_dim}")
        embedding_model = Emebeddings(model_name)
      
        # embedding_model.cuda()
        # max_length = embedding_model.model.tokenizer.model_max_length
        embeddings = embedding_model.encode(dataset)  
        sentence_embeddings.append(embeddings)
        # 截断每条文本的实现，在encode函数里面加入了truncate=True
        # embeddings = embedding_model.encode(texts[:100])
        bp()
        # 使用 UMAP 进行降维
    # print(len(sentence_embeddings))
    umap = UMAP()  # 降维到二维
    labels=[]
    for idx, model in enumerate(model_config):
        embeddings = sentence_embeddings[idx]
        labels.extend([idx] * len(embeddings)) 
    sentence_embeddings = np.vstack(sentence_embeddings)
    umap.plot(sentence_embeddings,labels)
      
    # # TODO 
        
