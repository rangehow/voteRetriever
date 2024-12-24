## 2024.12.24
  修改了保存特征那部分，使用https://huggingface.co/ISOISS/jina-embeddings-v3-tei上面教的使用模型的方法，提取outputs["hidden_states"]，并保存，Alibaba-NLP_gte-large-en-v1.5和ISOISS_jina-embeddings-v3-tei都没报错，现在直接运行run.py就可以直接调用modle里面的文件和函数，就可以，测量对齐分数还需要进一步修改，不知道分数奇不奇怪
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
