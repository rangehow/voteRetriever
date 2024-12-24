import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, AutoConfig
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
def auto_determine_dtype():
    """ automatic dtype setting. override this if you want to force a specific dtype """
    compute_dtype = torch.bfloat16 if check_bfloat16_support() else torch.float32
    torch_dtype = torch.bfloat16 if check_bfloat16_support() else torch.float32
    print(f"compute_dtype:\t{compute_dtype}")
    print(f"torch_dtype:\t{torch_dtype}")
    return compute_dtype, torch_dtype


def check_bfloat16_support():
    """ checks if cuda driver/device supports bfloat16 computation """
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        compute_capability = torch.cuda.get_device_capability(current_device)
        if compute_capability[0] >= 7:  # Check if device supports bfloat16
            return True
        else:
            return False
    else:
        return None
    

# 加载嵌入模型的类定义
class Embeddings:

    def __init__(self, model_name, device="cuda"):
        self.model = SentenceTransformer(model_name, trust_remote_code=True, device=device)

    def encode(self, texts, batch_size=32):
        return self.model.encode(
            texts, 
            batch_size=batch_size, 
            convert_to_numpy=True, 
            show_progress_bar=True, 
            truncate=True
        )

    def parameters(self):
        """ 提供模型参数计数 """
        return self.model._first_module().parameters()

    def __repr__(self) -> str:
        return str(self.model)


def load_llm(model_name, device="cuda"):
    print(f"Loading embedding model {model_name} ...")
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    print(f"Model {model_name} loaded successfully!")
    return model


# 加载分词器函数（用于 SentenceTransformer 不需要分词器配置）
def load_tokenizer(model_name):
    print(f"Loading tokenizer for {model_name} ...")
    model = AutoTokenizer.from_pretrained(model_name)
    print(f"Tokenizer for {model_name} loaded successfully!")
    return model
