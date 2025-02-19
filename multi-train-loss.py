import logging
import traceback
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from sentence_transformers import LoggingHandler, SentenceTransformer, evaluation, losses, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.similarity_functions import SimilarityFunction
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments


from collections.abc import Iterable

import torch
from torch import Tensor, nn

from sentence_transformers import SentenceTransformer
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)
# 多个教师模型
teacher_model_names = [
    "mixedbread-ai/mxbai-embed-large-v1",  
    "WhereIsAI/UAE-Large-V1",
    # "jinaai/jina-embeddings-v3",
    "avsolatorio/GIST-large-Embedding-v0",
    "BAAI/bge-large-en-v1.5",
    "Labib11/MUG-B-1.6"
]
teacher_models = [SentenceTransformer(name, trust_remote_code=True) for name in teacher_model_names]
# 教师模型，选取得是排名第八的dunzhang/stella_en_400M_v5模型 维度是8192
# teacher_model_name = "nvidia/NV-Embed-v2"
# teacher_model = SentenceTransformer(teacher_model_name,trust_remote_code=True)
#训练后的模型保存的路径
output_dir = "output/model-distillation-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# 选取了一个小的bert模型作为student来学习teacher的知识
student_model_name = "nreimers/TinyBERT_L-4_H-312_v2"
student_model = SentenceTransformer(student_model_name)

inference_batch_size = 8
train_batch_size = 8

logging.info("Load the AllNLI dataset")

nli_train_dataset = load_dataset("sentence-transformers/all-nli", "pair-score", split="train")#数量为942069
nli_eval_dataset = load_dataset("sentence-transformers/all-nli", "pair-score", split="dev") #19657

print(len(nli_train_dataset),len(nli_eval_dataset))
print(nli_train_dataset[0])

#将句子1和句子2合并
def combine_sentences(batch):
    return {"sentence": batch["sentence1"] + batch["sentence2"]}

#只保留sentence列
nli_train_dataset = nli_train_dataset.map(
    combine_sentences, batched=True, remove_columns=nli_train_dataset.column_names
)
nli_eval_dataset = nli_eval_dataset.map(combine_sentences, batched=True, remove_columns=nli_eval_dataset.column_names)

print(nli_train_dataset[0])

#将数据转换成pandas
def deduplicate(dataset):
    df = pd.DataFrame(dataset)
    df = df.drop_duplicates()#删除重复行
    return Dataset.from_pandas(df, preserve_index=False)#去掉原始索引，重新生成索引

#将训练数据和验证数据都这样转换
nli_train_dataset = deduplicate(nli_train_dataset)
nli_eval_dataset = deduplicate(nli_eval_dataset)

logging.info(nli_train_dataset)
logging.info("Load the STSB dataset")
#加载stsb数据集
stsb_eval_dataset = load_dataset("sentence-transformers/stsb", split="validation")
stsb_test_dataset = load_dataset("sentence-transformers/stsb", split="test")

logging.info(stsb_eval_dataset)
logging.info("Load the Wikipedia dataset")
wikipedia_train_dataset = load_dataset("sentence-transformers/wikipedia-en-sentences", split="train")
wikipedia_train_dataset_dict = wikipedia_train_dataset.train_test_split(test_size=5000)
wikipedia_train_dataset = wikipedia_train_dataset_dict["train"]
wikipedia_eval_dataset = wikipedia_train_dataset_dict["test"]
logging.info(wikipedia_train_dataset)
#concatenate_datasets是一个huggingface的函数，可以来合并两个数据集变成一个大的数据集，这里选取的是nli和wikipedia的训练集
train_dataset: Dataset = concatenate_datasets([nli_train_dataset, wikipedia_train_dataset])
#对于每个测试集，随机选取5000条数据集合并为一个大的数据集
eval_dataset: Dataset = concatenate_datasets(
    [nli_eval_dataset.select(range(5000)), wikipedia_eval_dataset.select(range(5000))]
)
#建立一个stsb的评估器，参数为两个句子，还有分数，还有衡量两个嵌入向量的相似性的方法，这里用的cos来衡量的
dev_evaluator_stsb = EmbeddingSimilarityEvaluator(
    sentences1=stsb_eval_dataset["sentence1"],
    sentences2=stsb_eval_dataset["sentence2"],
    scores=stsb_eval_dataset["score"],
    main_similarity=SimilarityFunction.COSINE,
    name="sts-dev",
)
logging.info("Teacher Performance")
#对教师模型进行评估

#如果学生模型的维度小于教师模型的维度，就要对教师模型的维度进行降维
for i in range(len(teacher_model_names)):
    dev_evaluator_stsb(teacher_models[i])
    if student_model.get_sentence_embedding_dimension() < teacher_models[i].get_sentence_embedding_dimension():
        logging.info("Student model has fewer dimensions than the teacher. Compute PCA for down projection")
        pca_sentences = nli_train_dataset[:20000]["sentence"] + wikipedia_train_dataset[:20000]["sentence"]
        pca_embeddings = teacher_models[i].encode(pca_sentences, convert_to_numpy=True)
        pca = PCA(n_components=student_model.get_sentence_embedding_dimension())
        pca.fit(pca_embeddings)

        # Add Dense layer to teacher that projects the embeddings down to the student embedding size
        dense = models.Dense(
            in_features=teacher_models[i].get_sentence_embedding_dimension(),
            out_features=student_model.get_sentence_embedding_dimension(),
            bias=False,
            activation_function=torch.nn.Identity(),
        )
        dense.linear.weight = torch.nn.Parameter(torch.tensor(pca.components_))
        teacher_models[i].add_module("dense", dense)

        logging.info(f"Teacher Performance with {teacher_models[0].get_sentence_embedding_dimension()} dimensions:")
      

def map_embeddings(batch):
    embeddings = teacher_models[0].encode(batch["sentence"], batch_size=inference_batch_size, show_progress_bar=False).tolist()
    return {"label": embeddings}


train_dataset = train_dataset.select(range(300000))
train_dataset=train_dataset.map(map_embeddings, batched=True, batch_size=50000)


eval_dataset.map(map_embeddings, batched=True, batch_size=50000)



class MSELoss(nn.Module):
    def __init__(self, model: SentenceTransformer) -> None:
        super().__init__()
        self.model = model
        self.loss_fct = nn.MSELoss()

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels:Tensor ) -> Tensor:
       
        loss = 0
        embeddings = self.model(sentence_features[0])["sentence_embedding"]
        for t_model in teacher_models:
            labels = t_model(sentence_features[0])["sentence_embedding"]
            loss = loss + self.loss_fct(embeddings, labels) 
        loss = loss/len(teacher_models) 
        return loss

    @property
    def citation(self) -> str:
        return

train_loss = MSELoss(model=student_model)
#评估数据
eval_sentences = eval_dataset["sentence"]
print("First sample in train_dataset:", train_dataset[0])

# 检查 eval_dataset 的第一个样本
print("First sample in eval_dataset:", eval_dataset[0])

loss_evaluator = [dev_evaluator_stsb]
for t in range(len(teacher_models)):
    loss_evaluator.append(evaluation.MSEEvaluator(eval_sentences, eval_sentences, teacher_model=teacher_models[t]))
#组合评估器
dev_evaluator = evaluation.SequentialEvaluator(loss_evaluator)

#训练参数
args = SentenceTransformerTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=train_batch_size,
    warmup_ratio=0.1,
    fp16=True,  
    bf16=False,  
    metric_for_best_model="eval_sts-dev_spearman_cosine",
    load_best_model_at_end=True,
    learning_rate=1e-4,
    
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=2000,
    save_total_limit=2,
    logging_steps=100,
    run_name="distillation-layer-reduction",  
)

#开始训练
trainer = SentenceTransformerTrainer(
    model=student_model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=train_loss,
    evaluator=dev_evaluator,
)
trainer.train()

test_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=stsb_test_dataset["sentence1"],
    sentences2=stsb_test_dataset["sentence2"],
    scores=stsb_test_dataset["score"],
    main_similarity=SimilarityFunction.COSINE,
    name="sts-test",
)
test_evaluator(student_model)

final_output_dir = f"{output_dir}/final"
student_model.save(final_output_dir)



if "/" in student_model_name:
    student_model_name = student_model_name.split("/")[-1]
if "/" in teacher_model_names[0]:
    teacher_model_names[0] = teacher_model_names[0].split("/")[-1]
repo_id = f"{student_model_name}-distilled-from-{teacher_model_names[0]}"
try:
    student_model.push_to_hub(repo_id)
except Exception:
    logging.error(
        f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, you can run "
        f"`huggingface-cli login`, followed by loading the model using `model = SentenceTransformer({final_output_dir!r})` "
        f"and saving it using `model.push_to_hub({repo_id!r})`."
    )