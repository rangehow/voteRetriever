import json
from module.data import load_data
from module.embeddings import Emebeddings
# from ipdb import set_trace as bp
# from module.dimension_reduction._umap import UMAP
# from loguru import logger

dataset_name = "allenai/c4"
if __name__ == "__main__":
    model_config = json.load(open("config/model.json"))
    dataset_config = json.load(open("config/data.json"))
    dataset = load_data(**dataset_config[dataset_name])
    sentence_embeddings=[]
    for model in model_config:
        model_name = model["model_name"]
        embedding_dim = model["embedding_dim"]
        logger.info(f"processing {model_name} with dim {embedding_dim}")
        model = Emebeddings(model_name)
        sentence_embeddings.append(model.encode(dataset))
        bp()
    umap = UMAP()
    # TODO 
    embeddings_2d = umap.plot(sentence_embeddings,label=...)
        
