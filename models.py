import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, AutoConfig

from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def get_models(modelset, model_path, modality='all'):
    assert modality in ['all', 'vision', 'language']

    # 定义模型路径
    if modelset == 'val':
        llm_models = [
            "huggyllama/llama-30b",
        ]
        lvm_models = [
            "vit_tiny_patch16_224.augreg_in21k",
        ]
        
    elif modelset == 'test':
        llm_models = [
            "huggyllama/llama-30b",
        ]
        lvm_models = []
        
    elif modelset == 'custom':
        llm_models = [
            "huggyllama/llama-30b",
        ]
        lvm_models = [
            "vit_tiny_patch16_224.augreg_in21k",
        ]
    else:
        raise ValueError(f"Unknown modelset: {modelset}")
    
    # 根据 modality 过滤模型
    if modality == "vision":
        llm_models = []
    elif modality == "language":
        lvm_models = []

    llm_models = [f"llm_model/{model}" for model in llm_models]
    lvm_models = [f"lvm_model/{model}" for model in lvm_models]

    return llm_models, lvm_models



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
    
    
def load_llm(llm_model_path, qlora=False, force_download=False, from_init=False):
    """ load huggingface language model """
    compute_dtype, torch_dtype = auto_determine_dtype()
    
    quantization_config = None
    if qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    if from_init:
        config = AutoConfig.from_pretrained(llm_model_path,
                                            device_map="auto",
                                            quantization_config=quantization_config,
                                            torch_dtype=torch_dtype,
                                            force_download=force_download,
                                            output_hidden_states=True,)
        language_model = AutoModelForCausalLM.from_config(config)
        language_model = language_model.to(torch_dtype)
        language_model = language_model.to("cuda" if torch.cuda.is_available() else "cpu")
        language_model = language_model.eval()
    else:      
        language_model = AutoModelForCausalLM.from_pretrained(
                llm_model_path,
                device_map="auto",
                quantization_config=quantization_config,
                torch_dtype=torch_dtype,
                force_download=force_download,
                output_hidden_states=True,
        ).eval()
    
    return language_model


def load_tokenizer(llm_model_path):
    """ setting up tokenizer. if your tokenizer needs special settings edit here. """
    tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
    
    if "huggyllama" in llm_model_path:
        tokenizer.pad_token = "[PAD]"        
    else:
        # pass 
        # tokenizer.add_special_tokens({"pad_token":"<pad>"})
        if tokenizer.pad_token is None:    
            tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    
    tokenizer.padding_side = "left"
    return tokenizer