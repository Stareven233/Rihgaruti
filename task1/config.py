import torch


LEN_OF_LINE = 100
SPLIT_FRAC = 0.8
BATCH_SIZE = 64
NUM_WORKER = 0

MODEL_PTH = 'model/lr0.001_batch64_epoch4_3.pth'
DATA_PTH = 'data/weibo_senti_100k.csv'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB_TO_INT_PTH = 'data/vocab_to_int.pkl'
VOCAB_SIZE_PTH = 'data/vocab_size.pkl'


ch_punc = r"！？｡。＂＃＄％＆＇（）＊＋，－―／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
pattern = r"(回复)?@.*?[\s:：]|[\s"  # 去掉@与空白部分
pattern += ch_punc+r"""!"#$%&'()*+,-./:;<=>?@\[\\\]^_`{|}~]"""  # 去标点符号等特殊字符
# 但实际上标点符号能够反映情绪，或许不去会好一些
