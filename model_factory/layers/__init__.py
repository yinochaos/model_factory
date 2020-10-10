from __future__ import absolute_import, division, print_function
#__all__ = ['seq2seq_cross_entropy_loss']
from model_factory.layers.attentions.multihead_attention import MultiHeadAttention, RelativeMultiHeadAttention
from model_factory.layers.embeddings.pos_embedding import TransformerEmbedding, RelativePositionalEmbedding
from model_factory.layers.recurrents.rnns import pBLSTM
from model_factory.layers.transformers.transformer import Transformer
from model_factory.layers.transformers.transformer_xl import TransformerXL
from model_factory.layers.transformers.longformer import Longformer
from model_factory.layers.transformers.bert import Bert, Albert
