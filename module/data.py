from datasets import load_dataset
from ipdb import set_trace as bp

def load_data(**kwargs):
    dataset = load_dataset(streaming=True, **kwargs)
    data = []
    for sample in dataset['train'].take(10000):
        data.append(sample["text"])
    return data