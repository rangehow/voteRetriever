import json
from module.data import load_data
from module.embeddings import Emebeddings
from ipdb import set_trace as bp
from module.dimension_reduction._umap import UMAP
from loguru import logger
import numpy as np
from module.measure_feature import extract_llm_features,load_and_average_features,compute_mean_embeddings,pad_embeddings
from module.measure_alignment import compute_alignment
import os
import argparse
from module.tasks import get_models
from tqdm import trange
import module.metrics
import torch
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision.models.feature_extraction import create_feature_extractor

from datasets import load_dataset
from pprint import pprint

dataset_name = "allenai/c4"
if __name__ == "__main__":
    model_config = json.load(open("config/model.json"))
    dataset_config = json.load(open("config/data.json"))
    print(dataset_config[dataset_name]["path"])
    # print(**dataset_config[dataset_name])
    dataset = load_data(**dataset_config[dataset_name])
    content_list = [item['text'] for item in dataset]
    parser = argparse.ArgumentParser()
    parser.add_argument("--force_download", action="store_true")
    parser.add_argument("--force_remake",   action="store_true")
    parser.add_argument("--num_samples",    type=int, default=1024)
    parser.add_argument("--batch_size",     type=int, default=4)
    parser.add_argument("--pool",           type=str, default='avg')
    parser.add_argument("--prompt",         action="store_true")
    parser.add_argument("--caption_idx",    type=int, default=0)
    parser.add_argument("--modelset",       type=str, default="val", choices=["val", "test"])
    parser.add_argument("--modality",       type=str, default="all", choices=["en", "zh", "all"])
    parser.add_argument("--output_dir",     type=str, default="./results/features")
    parser.add_argument("--input_dir",      type=str, default="./results/features")
    parser.add_argument("--output_dir_alignment",     type=str, default="./results/alignment")
    parser.add_argument("--precise",        action="store_true")
    parser.add_argument("--metric",         type=str, default="mutual_knn", choices=module.metrics.AlignmentMetrics.SUPPORTED_METRICS)
    parser.add_argument("--topk",           type=int, default=10)
    parser.add_argument("--modality_x",     type=str, default="all", choices=["en", "ch", "all"])
    parser.add_argument("--prompt_x",       action="store_true")
    parser.add_argument("--pool_x",         type=str, default=None, choices=['avg', 'cls'])
    parser.add_argument("--modality_y",     type=str, default="all", choices=["en", "ch", "all"])
    parser.add_argument("--prompt_y",       action="store_true")
    parser.add_argument("--pool_y",         type=str, default=None, choices=['avg', 'cls'])
    args = parser.parse_args()
    print(args.modality)
    llm_models = get_models(args.modelset,args.modality)
    extract_llm_features(llm_models, content_list, args)
    directory_path = '/mnt/ljy/metb/voteRetriever/results/features'

    filenames = [os.path.join(directory_path, fname) for fname in os.listdir(directory_path)]
    all_feats = load_and_average_features(filenames)
    
    target_dim = 1024
    features = pad_embeddings(all_feats, target_dim)
    mean_features  = compute_mean_embeddings(features)
    print( mean_features )
    print(mean_features.shape)
    umap = UMAP()
    
    labels=[]
    for idx in range(len(filenames)):
        labels.extend([idx]) 
    print(len(labels))
    umap.plot(mean_features, labels)
    # if not args.precise:
    #     torch.set_float32_matmul_precision('high')
    #     torch.backends.cuda.matmul.allow_tf32 = True
    #     torch.backends.cudnn.allow_tf32 = True
    #     torch.backends.cudnn.benchmark = True
    # save_path = module.utils.to_alignment_filename(
    #         args.output_dir,
    #         args.modality_x, args.pool_x, args.prompt_x,
    #         args.modality_y, args.pool_y, args.prompt_y,
    #         args.metric, args.topk
    # )

    # if os.path.exists(save_path) and not args.force_remake:
    #     print(f"alignment already exists at {save_path}")
    #     exit()

    # models_x = llm_models 
    # models_y = llm_models 
  
    # models_x_paths = [module.utils.to_feature_filename(args.input_dir, m) for m in models_x]
    # models_y_paths = [module.utils.to_feature_filename(args.input_dir, m) for m in models_y]
    # print(models_x_paths)
    # for fn in models_x_paths + models_y_paths:
    #     assert os.path.exists(fn), fn
    
    # print(f"metric: \t{args.metric}")
    # if 'knn' in args.metric:
    #     print(f"topk:\t{args.topk}")
    
    # print(f"models_x_paths:")    
    # pprint(models_x_paths)
    # print("\nmodels_y_paths:")
    # pprint(models_y_paths)
    
    # print('\nmeasuring alignment')
    # os.makedirs(args.output_dir_alignment, exist_ok=True)
    # alignment_scores, alignment_indices = compute_alignment(models_x_paths, models_y_paths, args.metric,args.topk)

    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # np.save(save_path, {"scores": alignment_scores, "indices": alignment_indices})
    # print({"scores": alignment_scores, "indices": alignment_indices})
    # print(f"saved to {save_path}")
        
