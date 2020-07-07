# Language-agnostic BERT Sentence Embedding (LaBSE)

Convert the original tfhub weights to the BERT format.

## LaBSE

- Paper: https://arxiv.org/abs/2007.01852
- TFHUB: https://tfhub.dev/google/LaBSE/1

Original Introduction:

> We adapt multilingual BERT to produce language-agnostic sentence embeddings for 109 languages. %The state-of-the-art for numerous monolingual and multilingual NLP tasks is masked language model (MLM) pretraining followed by task specific fine-tuning. While English sentence embeddings have been obtained by fine-tuning a pretrained BERT model, such models have not been applied to multilingual sentence embeddings. Our model combines masked language model (MLM) and translation language model (TLM) pretraining with a translation ranking task using bi-directional dual encoders. The resulting multilingual sentence embeddings improve average bi-text retrieval accuracy over 112 languages to 83.7%, well above the 65.5% achieved by the prior state-of-the-art on Tatoeba. Our sentence embeddings also establish new state-of-the-art results on BUCC and UN bi-text retrieval.


## Download

The converted weights can be downloaded at: 

> 链接: https://pan.baidu.com/s/17qUdDSrPhhNTvPnEeI56sg 提取码: p52d

We can load it with [bert4keras](https://github.com/bojone/bert4keras):
```python
from bert4keras.backend import keras
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
import numpy as np

config_path = '/root/kg/bert/labse/bert_config.json'
checkpoint_path = '/root/kg/bert/labse/bert_model.ckpt'
dict_path = '/root/kg/bert/labse/vocab.txt'

tokenizer = Tokenizer(dict_path)
model = build_transformer_model(config_path, checkpoint_path, with_pool='linear')

# 编码测试
token_ids, segment_ids = tokenizer.encode(u'语言模型')

print('\n ===== predicting =====\n')
print(model.predict([np.array([token_ids]), np.array([segment_ids])]))
```

## Contact
- https://kexue.fm
