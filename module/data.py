from datasets import load_dataset,load_from_disk,Dataset
from ipdb import set_trace as bp
import os

def load_data(**kwargs):
    cache_dir = os.path.join("data", kwargs.get("path", None).rsplit('/')[-1])
    num_rows = kwargs.pop("num_rows", None)
    text_key = kwargs.pop("text_key", "text")
    if os.path.exists(cache_dir):
        dataset = load_from_disk(cache_dir)
    
    else:
        iterable_dataset = load_dataset(streaming=True, **kwargs)['train'].take(num_rows)
        dataset = Dataset.from_dict({
            f'{text_key}': [sample[f'{text_key}'] for sample in iterable_dataset]
        })
        dataset.save_to_disk(cache_dir)
        
    data = []
    for sample in dataset:
            data.append(sample[f'{text_key}'])

    return data
