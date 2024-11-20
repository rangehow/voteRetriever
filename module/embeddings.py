from sentence_transformers import SentenceTransformer

class Emebeddings:

    def __init__(self,model_name) -> None:
        self.model = SentenceTransformer(model_name,trust_remote_code=True,device='cuda')
        

    def encode(self, texts):
        return self.model.encode(texts,batch_size=32,convert_to_numpy=True,show_progress_bar=True,truncate=True)
    
    def __repr__(self) -> str:
        return str(self.model)




