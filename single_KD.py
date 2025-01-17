import logging
import traceback
from datetime import datetime

import pandas as pd
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from sklearn.decomposition import PCA

from sentence_transformers import LoggingHandler, SentenceTransformer, evaluation, losses, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.similarity_functions import SimilarityFunction
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

#### 日志记录
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)

# 教师模型，选取得是排名第八的dunzhang/stella_en_400M_v5模型 维度是8192
teacher_model_name = "nvidia/NV-Embed-v2"
teacher_model = SentenceTransformer(teacher_model_name,trust_remote_code=True)

#训练后的模型保存的路径
output_dir = "output/model-distillation-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# 选取了一个小的bert模型作为student来学习teacher的知识
student_model_name = "intfloat/e5-small"
student_model = SentenceTransformer(student_model_name)

inference_batch_size = 64
train_batch_size = 64

logging.info("Load the AllNLI dataset")

#加载数据，数据含有三个标签，数据集下面这个样子
# """
# Dataset({
#     features: ['premise', 'hypothesis', 'label'],
#     num_rows: 942069
# })
# """
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
dev_evaluator_stsb(teacher_model)
#如果学生模型的维度小于教师模型的维度，就要对教师模型的维度进行降维
if student_model.get_sentence_embedding_dimension() < teacher_model.get_sentence_embedding_dimension():
    logging.info("Student model has fewer dimensions than the teacher. Compute PCA for down projection")
    #选取nli和wikipedia数据集的前两万条 对其进行encode，然后使用这些嵌入向量对pca进行训练
    pca_sentences = nli_train_dataset[:20000]["sentence"] + wikipedia_train_dataset[:20000]["sentence"]
    pca_embeddings = teacher_model.encode(pca_sentences, convert_to_numpy=True)
    pca = PCA(n_components=student_model.get_sentence_embedding_dimension())#降低的维度为学生模型的嵌入维度
    pca.fit(pca_embeddings)
    #创建一个全连接层，输入是教师模型的嵌入维度，目标输出是学生模型的嵌入维度
    dense = models.Dense(
        in_features=teacher_model.get_sentence_embedding_dimension(),
        out_features=student_model.get_sentence_embedding_dimension(),
        bias=False,
        activation_function=torch.nn.Identity(),
    )
    #pca.components_是[n_components, n_features] 的形状，n_components 是即学生模型的维度，n_features 是教师模型的维度
    dense.linear.weight = torch.nn.Parameter(torch.tensor(pca.components_))
    teacher_model.add_module("dense", dense)

    logging.info(f"Teacher Performance with {teacher_model.get_sentence_embedding_dimension()} dimensions:")
    #评估教师模型性能
    dev_evaluator_stsb(teacher_model)

#获得训练和验证数据的嵌入向量
def map_embeddings(batch):
    return {
        "label": teacher_model.encode(
            batch["sentence"], batch_size=inference_batch_size, show_progress_bar=False
        ).tolist()
    }

train_dataset = train_dataset.select(range(200000))
train_dataset = train_dataset.map(map_embeddings, batched=True, batch_size=50000)
#将数据保存，便于后续使用
train_dataset.save_to_disk("datasets/distillation_train_dataset")
eval_dataset = eval_dataset.map(map_embeddings, batched=True, batch_size=50000)
#训练损失为均方误差
train_loss = losses.MSELoss(model=student_model)
#评估数据
eval_sentences = eval_dataset["sentence"]
#创建mse评估器
dev_evaluator_mse = evaluation.MSEEvaluator(eval_sentences, eval_sentences, teacher_model=teacher_model)
#组合评估器
dev_evaluator = evaluation.SequentialEvaluator([dev_evaluator_stsb, dev_evaluator_mse])

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
    save_steps=500,
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

#保存模型到huggingface上面

if "/" in student_model_name:
    student_model_name = student_model_name.split("/")[-1]
if "/" in teacher_model_name:
    teacher_model_name = teacher_model_name.split("/")[-1]
repo_id = f"{student_model_name}-distilled-from-{teacher_model_name}"
try:
    student_model.push_to_hub(repo_id)
except Exception:
    logging.error(
        f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, you can run "
        f"`huggingface-cli login`, followed by loading the model using `model = SentenceTransformer({final_output_dir!r})` "
        f"and saving it using `model.push_to_hub({repo_id!r})`."
    )