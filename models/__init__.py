from models.gpt import GPTModel
from models.lstm import LSTMModel
from models.lstm_attention import LSTMModelWithAttention
from models.markov import MarkovModel
from models.multi_gpt import MultiGPTModel
from models.multi_lstm import MultiLSTMModel
from models.multi_lstm_attention import MultiLSTMAttentionModel
from models.multi_transformer import MultiTransformerModel
from models.transformer import TransformerModel

MODEL_REGISTRY = {
    "MarkovModel": MarkovModel,
    "LSTMModel": LSTMModel,
    "LSTMModelWithAttention": LSTMModelWithAttention,
    "TransformerModel": TransformerModel,
    "GPTModel": GPTModel,
    "MultiLSTMModel": MultiLSTMModel,
    "MultiLSTMAttentionModel": MultiLSTMAttentionModel,
    "MultiTransformerModel": MultiTransformerModel,
    "MultiGPTModel": MultiGPTModel
}
