"""Model
0. Embedding
1. Attention
2. MultiHead Attention
3. Encoder
4. Full class (BERT)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math 
from src.relative_position import RelativePositionalEmbedding

# We have 3 embedding: (1) Token: for each word meaning, 
# (2) Position: for each word position, (3) segment: for each word, what sentence it belongs to

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class SinCosPositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, maxlen: int, dropout: int, device):
        super(SinCosPositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0,emb_size, 2) * math.log(10000) /emb_size)
        pos = torch.arange(0,maxlen).reshape(maxlen, 1) 
        self.pos_emb = torch.zeros((maxlen,emb_size))
        self.pos_emb[:, 0::2] = torch.sin(pos * den) # this needs to be both of same size; emb_size needs to be even
        self.pos_emb[:, 1::2] = torch.cos(pos * den)
        self.pos_emb = self.pos_emb.unsqueeze(0).to(device) # (0) to make place for batch at first

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('sin_pos_emb', self.pos_emb)
        
    def forward(self, seq_len):
        return self.pos_emb[:, :seq_len, :]

class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size, n_heads, max_relative_position, dropout, device=device, pos_emb='sin', maxlen=1024):
        super(BERTEmbedding, self).__init__()
        self.token_emb = nn.Embedding(vocab_size, emb_size)
        self.attn_pos_emb = nn.Embedding(vocab_size, emb_size) # This can be Sinusiodal Positional encoding, numeric positions, relative positional...
        self.segment_emb = nn.Embedding(vocab_size, emb_size)
        self.pos_emb = pos_emb
        self.device = device
        # Integration 1: 
        self.relative_pos_emb = RelativePositionalEmbedding(emb_size, n_heads, max_relative_position, dropout, device)
        self.sin_pos_encoding = SinCosPositionalEncoding(emb_size, maxlen, dropout, device)
    def forward(self, x, atten_mask):
        sentence_size = x.size(1)
        tok_emb = self.token_emb(x)
         # Integration 2: 
        if self.pos_emb == 'relative':
            pos_emb = self.relative_pos_emb(tok_emb, tok_emb, tok_emb, atten_mask)

        elif self.pos_emb == 'attn': 
            pos_emb_tensor = torch.arange(sentence_size, dtype=torch.long).to(self.device) # We want tensor by sentence size
            pos_emb_tensor = pos_emb_tensor.expand_as(x) # to make them same sizes for broadcast of addition
            pos_emb = self.attn_pos_emb(tok_emb)
        
        elif self.pos_emb == 'sin':
            pos_emb = self.sin_pos_encoding(seq_len=sentence_size) # BUG: this is has no batch info, is this expected? given it's indep of seq and only needs its length?

        # Note that the input is composed of 2 sentences. (we prepared the data this way)
        seg_emb_tensor = torch.zeros(x.size()).to(device) # to make a tensor that will be then filled by 0/1 denoting sentences
        seg_emb_tensor[:, sentence_size // 2 + 1:] = 1 # we know half of sentence means one sentence as we made all sentences of one common size
        total_input_emb =  tok_emb + self.segment_emb(seg_emb_tensor.long()) + pos_emb # + 
        # shape: batch_size, seq_len, emb_dim
        return total_input_emb

class SelfAttention(nn.Module):
    def __init__(self, emb_size):
        super(SelfAttention, self).__init__() 
        self.emb_size = emb_size
        self.linear = nn.Linear(emb_size, emb_size)
    
    def forward(self, q, k, v, mask=None):
        # 1. linear layer
        q, k, v = self.linear(q), self.linear(k), self.linear(v)
        # 2. Matrix Multiplication (Q*K)
        # shape: bsz, seq_len, seq_len
        qk = torch.matmul(q, k.transpose(-2, -1)) 
        # 3. Scaling
        qk_scale = qk / math.sqrt(self.emb_size) 
        # 4. Ignore padding (not attend to them)
        if mask != None:
            # BUG: from here starts the bug; as for 3rd sentence: all false "-inf" (no padding)
            qk_scale = qk_scale.masked_fill(mask==1, float('-inf')) # This was a BUG as we refer to what's padded by True (1) while mask was put to 0, thus ignoring everything
        # 5. Softmax
        qk_soft = F.softmax(qk_scale, dim=-1) # shape: bs, emb, emb
        # 6. Multiply by Value
        qkv = torch.matmul(qk_soft, v) # shape: bsz, seq_len, emb_size

        return qkv

# Multihead is just many parallel self attention; as filters in CNN
# meaning that for each query word; we'll look to different (various) context words
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, emb_size):
        super(MultiHeadAttention, self).__init__()
        # Make n_head self_attention (not divide), initialize them then use in forward
        self.heads = nn.ModuleList([SelfAttention(emb_size) for _ in range(n_heads)])
        self.linear = nn.Linear(emb_size*n_heads, emb_size) # input is multiplied by numb of heads (for concatenation)
        self.norm = nn.LayerNorm(emb_size)

    def forward(self, x, atten_pad_mask):
        # call self attention to make list of multiheads
        multihead = [head(x, x, x, atten_pad_mask) for head in self.heads] # list: 8 (8 heads; each is self_atten_embedding; qkv coming from summation of 3 embedding)
        # Concatenate them in one
        multihead_cat = torch.cat(multihead, dim=-1) # shape: bsz, seq_len, n_head*emb (concatenate over the emb_dim)
        # Then make linear and norm
        multihead_norm = self.norm(self.linear(multihead_cat)) # normalize after linearly transforming to have shape: bsz, seq_len, emb
        return multihead_norm

class EncoderLayer(nn.Module):
    def __init__(self, n_heads, emb_size, dense_dim, dropout_rate):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(n_heads, emb_size)
        self.norm = nn.LayerNorm(emb_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(emb_size, dense_dim), 
            nn.Dropout(dropout_rate),
            nn.GELU(),
            nn.Linear(dense_dim, emb_size),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x, atten_pad_mask):
        # 1. Attention
        attention = self.multi_head_attention(x, atten_pad_mask) 
        # 2. Add & Norm
        x_atten_and_norm = self.norm((attention + x))
        # 3. FeedForward
        x_ff = self.feed_forward(x_atten_and_norm)
        # 4. Add & Norm
        x_ff_and_norm = self.norm((x_atten_and_norm + x_ff))
        return x_ff_and_norm

class Encoder(nn.Module):
    def __init__(self, n_layers, *layer_args):
        super(Encoder, self).__init__()
        self.encoder_layers = nn.ModuleList([EncoderLayer(*layer_args) for _ in range(n_layers)]) 

    def forward(self, x, atten_pad_mask): 
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, atten_pad_mask) # new embedding coming from the encoder layer after making attention on embedding of input (token/pos/segment)
        return x

class BERTModel(nn.Module):
    """Building the BERT Model. Mainly Embedding and Encoder => Features. 
       Linear layers are for projection on token_prediction and for classifying NSP or not
    """
    def __init__(self, vocab_size, emb_size, n_layers, n_heads, dense_dim, dropout_rate, max_relative_position=10):
        super(BERTModel, self).__init__()
        self.embedding = BERTEmbedding(vocab_size, emb_size, n_heads, max_relative_position, dropout_rate, device)
        self.encoder = Encoder(n_layers, n_heads, emb_size, dense_dim, dropout_rate) 
        self.mask_prediction_layer = nn.Linear(emb_size, vocab_size) # we want to project over the vocab space
        # As we won't use CCE then we need to explicitly make softmax
        self.softmax = nn.LogSoftmax(dim=-1)
        self.nsp_prediction_layer = nn.Linear(emb_size, 2)

    def forward(self, x, atten_pad_mask):
        embedding = self.embedding(x, atten_pad_mask) # shape: bsz, seq_len, emb_size
        encoding = self.encoder(embedding, atten_pad_mask)
        mask_prediction = self.mask_prediction_layer(encoding) # This serves as logits

        first_word = encoding[:, 0, :] 
        nsp_prediction = self.nsp_prediction_layer(first_word) # This isn't 0-1 but floats  + and - 

        return self.softmax(mask_prediction), nsp_prediction
