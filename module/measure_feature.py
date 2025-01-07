import gc
import os
import argparse
from tqdm import trange
import torch
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision.models.feature_extraction import create_feature_extractor
from datasets import load_dataset
from module.models import load_llm, load_tokenizer
import module.utils 
from module.data import load_data
from module.tasks import get_models
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
import os
import gc
from tqdm import trange
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
def extract_llm_features(filenames, texts, args):
    for llm_model_name in filenames[::-1]:
        save_path = module.utils.to_feature_filename(
            args.output_dir, llm_model_name
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"Processing:\t{llm_model_name}")
        print(f"Save path: \t{save_path}")

        if os.path.exists(save_path) and not args.force_remake:
            print("File exists. Skipping.")
            continue
        
        language_model = load_llm(llm_model_name)
        tokenizer = load_tokenizer(llm_model_name)
        language_model.to('cuda')


        llm_feats, scores = [], []

        for i in range(0, len(texts), args.batch_size):
            batch_texts = texts[i:i + args.batch_size]
            batch_dict = tokenizer(batch_texts, max_length=8192, padding=True, truncation=True, return_tensors='pt').to('cuda')
            outputs = language_model(**batch_dict,output_hidden_states=True)
            if args.pool == 'avg':
                    feats = torch.stack(outputs["hidden_states"]).permute(1, 0, 2, 3)
                    mask = batch_dict["attention_mask"].unsqueeze(-1).unsqueeze(1)
                    feats = (feats * mask).sum(2) / mask.sum(2)
            elif args.pool == 'last':
                    feats = [v[:, -1, :] for v in outputs["hidden_states"]]
                    feats = torch.stack(feats).permute(1, 0, 2) 
            else:
                    raise NotImplementedError(f"unknown pooling {args.pool}")
            llm_feats.append(feats.cpu())
        import numpy as np
# 在将张量转换为 NumPy 数组之前，使用 .detach()
        llm_feats_float32 = [feat.detach().float() for feat in llm_feats]
        llm_feats_array = np.array([feat.numpy() for feat in llm_feats_float32])

            # 打印 shape
        print(llm_feats_array.shape)

  
        save_dict = {
            "feats": torch.cat(llm_feats).cpu()
            
        }
        torch.save(save_dict, save_path)
        print(f"Saved to {save_path}")
        
        # 清理缓存
        del language_model, llm_feats, feats
        torch.cuda.empty_cache()
        gc.collect()

    return
def load_and_average_features(filenames):
    all_feats = []
    
    for filename in filenames:
        if os.path.exists(filename):
            print(f"Loading features from {filename}")
            feature_data = torch.load(filename)
            feats = feature_data["feats"]  
            all_feats.append(feats)
        else:
            print(f"File not found: {filename}")

    return all_feats


def compute_mean_embeddings(features):
    """
    计算每组嵌入的均值
    :param features: List[Tensor or list]，每组嵌入可能是列表或张量
    :return: Tensor，形状为 [n_models, D]，每组嵌入的均值
    """
    mean_features = []
    for feat in features:
        # 如果 feat 是嵌套列表，将其转换为二维列表或张量
        if isinstance(feat, list):
            feat = torch.tensor(feat)  # 转换为 PyTorch 张量

        # 确保 feat 是二维张量
        if feat.dim() == 1:
            feat = feat.unsqueeze(0)  # 如果是 1D 张量，扩展为 2D

        # 计算均值
        mean_feat = torch.mean(feat, dim=0)  # 对每组嵌入求均值
        mean_features.append(mean_feat)
    
    return torch.stack(mean_features)  # 合并成一个大张量



def pad_embeddings(features, target_dim):
    """
    对嵌入进行补零，使其对齐到 target_dim
    :param features: List[Tensor]，每个张量表示一组嵌入
    :param target_dim: int，目标维度
    :return: List[Tensor]，补零后的嵌入
    """
    padded_features = []
    for feat in features:
        if feat.shape[1] < target_dim:  # 如果维度不足，补零
            padding = torch.zeros(feat.shape[0], target_dim - feat.shape[1], device=feat.device)
            feat = torch.cat((feat, padding), dim=1)
        padded_features.append(feat)
    return padded_features

