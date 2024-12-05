import logging
from transformers import AutoModel, AutoTokenizer
import os
os.environ['HF_HOME'] = '/root/autodl-tmp/cache/'
# 启用日志
logging.basicConfig(level=logging.INFO)

# 下载模型和分词器
model_name = "sgugger/glue-mrpc"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
