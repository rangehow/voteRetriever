from mteb import MTEB
from sentence_transformers import SentenceTransformer
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Define the sentence-transformers model name
model_name = "/mnt/ljy/teacher_student/output/model-distillation-2025-01-16_21-29-24/final"

model = SentenceTransformer(model_name,trust_remote_code=True)
evaluation = MTEB(tasks=["Banking77Classification"])
results = evaluation.run(model, output_folder=f"results/{model_name}")