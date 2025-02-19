## 2025.2.18
使用30万条数据混合平均多教师模型的MSE损失，得到的正确率达到0.611，用单个模型训练的时候，比0.61高，可以达到0.67（使用的是第一种蒸馏方法，选取一个小的学生模型的那个）
## 2025.2.17
对于多教师模型，混合loss修改成功

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

这里的sentence_features是sentencetransformers库里的trainer.py里面的collect_features这个函数获得的，sentence_features[0]代表的是input_ids，embeddings = self.model(sentence_features[0])["sentence_embedding"]这里获得的是学生模型的嵌入，labels获得的是每个教师模型的嵌入，（实际上传入的labels没用上），然后用loss_fct计算MSE损失，
然后将每个教师模型嵌入和学生嵌入的损失求和，然后再平均求一下，进行训练，但是在最开始训练的时候，准确率下降（比student_model，准确率0.54），大概0.4，然后用五万条数据训练1轮之后，准确率上升到0.55，注意：使用两个k=2教师模型训练的时候，loss离谱的低，准确率却离谱的低，所以猜测，只有多个教师模型训练的时候才可以
## 2025.2.8
平均得嵌入那里，我无法运行任意前k个模型，out of memory，用了学长保存在embeddings里面的嵌入，看了代码发现是平均后的，形状只有（1，维度），可能还得学长运行一下
然后用了kmeans的方法对嵌入聚为1类，对小模型进行训练，观察loss非常小，训练之后，正确率只有0.28，失败。
然后用了另一种多教师模型蒸馏的方法，保留其中一个教师模型teacher_model_1的部分层（我保留了8层），用另一个教师模型的嵌入去训练，10万条数据的时候，正确率能达到0.834(teacher_model_1的正确率为0.87）evaluation_time减少了，正确率相差还好，这种应该也算模型蒸馏吧，k个模型的时候，可以混合多个模型层，因为他们的结构相似，我试了两个模型部分层混合，也可以，20万条数据训练的时候，对于不同模型混合层和一个模型部分层的正确率差不多，0.73。
## 2025.1.21
对于多teacher模型蒸馏，在维度是1024的模型上选了几个模型，使用对每个模型encode后得到的嵌入进行求均值，进行训练，在Banking77Classification任务上，使用mteb测评，得到分数（0.61987）比student（0.540097）要高，但是没有单个teacher模型对student蒸馏的分数高（0.658182），分析原因，可能是因为模型簇里面有分数高和分数低的模型，平均后分数低了，或者模型不够多，平均后的并不能代表嵌入的质心，不知道什么原因，然后随机选取了不是簇内的模型，还在等待结果。
看到了一篇论文，讲述多教师模型蒸馏的训练过程，运用了混合损失，正在复现，大概1.23之前完成（完成了，但是错了）。
## 2025.1.17
已经完成在Banking77Classification任务上，使用mteb测评，分数相对于之前提高了，但是分数还是低于teacher，接下来对多教师模型修改代码，还没改出来，正在学习中
## 2024.12.25
  修改了计算分数部分，这样计算分数和提取特征就不会因为模型不同报错了，得到的分数正常多了，模型自己和自己之间的相似性分数为1，Alibaba-NLP_gte-large-en-v1.5和ISOISS_jina-    embeddings-v3-tei之间的分数为0.55299997，下一步就是对收集好的一些模型跑一下，都相互计算出分数，然后画图
## 2024.12.24
  修改了保存特征那部分，使用https://huggingface.co/ISOISS/jina-embeddings-v3-tei 上面教的使用模型的方法，提取outputs["hidden_states"]，并保存，Alibaba-NLP_gte-large-en-v1.5和ISOISS_jina-embeddings-v3-tei都没报错，现在直接运行run.py就可以直接调用modle里面的文件和函数，就可以，测量对齐分数还需要进一步修改，不知道分数奇不奇怪
## 2024.12.22
  原来的代码是使用的类似llama这种模型，对数据进行tokenizer，然后得到hidden_states作为模型的特征，存下来计算后续的相似分数，但是现在使用的是嵌入模型，我直接将encode后的embedding当作特征存储下来，不知道这样行不行，然后计算相关性分数，但是得到的分数非常奇怪，然后再去仔细跑一下提取特征的代码，看一看，就报错了,还在改
## 2024.12.3
  代码跑通了，模型保存在tasks.py中，数据集使用的是minhuh/prh，先运行platonic-rep/extract_features.py提取特征，保存为platonic-rep/results/features/minhuh/prh/wit_1024/huggyllama_llama-30b_pool-avg.pt这样的文件，然后运行platonic-rep/measure_alignment.py进行对齐分数的计算，将每一对模型的对齐分数和相对应的索引进行保存，保存为platonic-rep/results/alignment/minhuh/prh/val/language_pool-avg_prompt-False_vision_pool-cls_prompt-False/mutual_knn_k10.npy
  试了一个例子，是llama-30b和vit_tiny_patch16_224之间的对齐分数{'scores': array([[0.12714845]]), 'indices': array([[[11., 27.]]])}
## 2024.11.28
  写了一个柏拉图表征假说的代码分析，主要有提取特征和计算对齐分数两部分。在https://zhuanlan.zhihu.com/p/9622855585 具体公式在论文附录A
## 2024.11.24
  写了一个柏拉图表征假说的论文解读（表征趋同部分，以及图片的含义）在https://zhuanlan.zhihu.com/p/8068657917上
## 2024.11.23
 去MTEB Leaderboard榜单上按照model.json格式添加了一些模型
## 2024.11.20
 修改下载和读数据的形式，适合用在大多数hf数据上
  
  对于文本过长，截断部分，其实不用手动截断，sentence transformer里面有这个参数
  
  sentence embedding和labels是这样的：
  
  有m个模型（这里是0，1，2）
 
 sentences=[Embedding1,...,Embeddingm],Embeddingi是N*dm，就是[[N*dm],[N*dm],[N*dm]]
  然后经过np.vstack变成[3N*dm]
  
  labels=[0,...0,1,...1,2,...2]
## 2024.11.19 
  下载数据集遇到困难，使用huggingface-cli 下载了一个json文件，然后选择100条进行embedding，然后encode部分，由于文本过长报错，进行手动截断
