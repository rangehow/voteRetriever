

def get_models(modelset, modality):
    
    if modelset == 'val':
        if modality == 'en':
            llm_models = [
                # "bigscience/bloomz-560m",
                # "bigscience/bloomz-1b1",
                # "bigscience/bloomz-1b7",
                # "bigscience/bloomz-3b",
                # "bigscience/bloomz-7b1",
                # "openlm-research/open_llama_3b",
                # "openlm-research/open_llama_7b",
                # "openlm-research/open_llama_13b",
                "Alibaba-NLP/gte-large-en-v1.5",
                "ISOISS/jina-embeddings-v3-tei",
                # "dunzhang/stella_en_400M_v5",
                # "huggyllama/llama-65b",
            ]
        elif modality == 'zh':
            llm_models = [
                # "bigscience/bloomz-560m",
                # "bigscience/bloomz-1b1",
                # "bigscience/bloomz-1b7",
                # "bigscience/bloomz-3b",
                # "bigscience/bloomz-7b1",
                # "openlm-research/open_llama_3b",
                # "openlm-research/open_llama_7b",
                # "openlm-research/open_llama_13b",
                # "huggyllama/llama-7b",
                "ISOISS/jina-embeddings-v3-tei",
                "huggyllama/llama-30b",
                # "huggyllama/llama-65b",
            ]
        
    elif modelset == 'test':
        llm_models = [
            # "allenai/OLMo-1B-hf",
            # "allenai/OLMo-7B-hf", 
            # "google/gemma-2b",
            # "google/gemma-7b",
            # "mistralai/Mistral-7B-v0.1",
            # "mistralai/Mixtral-8x7B-v0.1",
            # # "mistralai/Mixtral-8x22B-v0.1",
            # "NousResearch/Meta-Llama-3-8B",
            # "NousResearch/Meta-Llama-3-70B",
        ]
    elif modelset == 'all':
        llm_models = [
            # "bigscience/bloomz-560m",
            # "bigscience/bloomz-1b1",
            # "bigscience/bloomz-1b7",
            # "bigscience/bloomz-3b",
            # "bigscience/bloomz-7b1",
            # "openlm-research/open_llama_3b",
            # "openlm-research/open_llama_7b",
            # "openlm-research/open_llama_13b",
            # "huggyllama/llama-7b",
            # "huggyllama/llama-13b",
            "huggyllama/llama-30b",
            "huggyllama/llama-65b",
            # "allenai/OLMo-1B-hf",
            # "allenai/OLMo-7B-hf", 
            # "google/gemma-2b",
            # "google/gemma-7b",
            # "mistralai/Mistral-7B-v0.1",
            # "mistralai/Mixtral-8x7B-v0.1",
            # # "mistralai/Mixtral-8x22B-v0.1", # was too big so did not use
            # "NousResearch/Meta-Llama-3-8B",
            # "NousResearch/Meta-Llama-3-70B",
        ]
    else:
        raise ValueError(f"Unknown modelset: {modelset}")
    return llm_models
