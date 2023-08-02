import torch
import math
from torch import nn
import torch.nn.functional as F

# Objective: learn embedding vector for each "relative" position
# Steps: (1) Identify matrix of possible relatives (sent_len, sent_len) "clamped values" 
# (2) Identify embedding vector with possible vocabs of relatives (vocab, emb)
# (3) Input 1 into 2 to have position matrix to learn on (PER SENTENCE)
class RelativePosition(nn.Module): # 2 stages: embeddings with vocab of available relativities + tokens of indices/relatives to learn each relative position
    def __init__(self, max_relative_position, head_dim, device):
        super().__init__()
        self.head_dim = head_dim
        self.max_relative_position = max_relative_position
        # embedding: (vocab x emb_dim): vocab here is of size --> max * 2 (left/right) + 1 (relation with myself). e.g. (max = 3; "3*2+1 = 7")
        # embedding table is table of random embeddings that learns by time. we only specify the size of available vocab here
        # Embedding we have 512 vector as embedding representing each token in vocab (Initialization)
        # When we use the embedding (pass sentence to it) --> we get the corresponding 512 vector of each token from sentence (vocab) then we change this by learning until its suitable for all sentences
        self.embedding_table = nn.Embedding(max_relative_position*2 + 1, head_dim)
        # nn.embedding (num_possible_emb_dict, emb_dim), Out shape: (*, emb_dim) --> where * is same as input shape
        self.device = device
    def forward(self, len_q, len_k): # for self attention (q/k/v) have same length as sequence, but this isn't always the case
        possible_relatives_1d_q = torch.arange(len_q)
        possible_relatives_1d_k = torch.arange(len_k)
        #  Make row matrix - column matrix [0, 1, 2, 3] - [[0, 1, 2, 3]] --> subtract full row from each token in column
        # q is fixed in 2nd position (for its the query in hand?)
        possible_relatives_2d = possible_relatives_1d_k[None, :] -  possible_relatives_1d_q[:, None] # (Instead of None: put 1)
        clamped_relatives_2d = torch.clamp(possible_relatives_2d, -self.max_relative_position, self.max_relative_position) # clamp that no min is less than specified min and same for max (relativity won't differ beyond that)
        # shape: len_q x len_k (self attention: seq_len x seq_len)
        # To make all positives (no -ve input to network)
        clamped_relatives_positive = clamped_relatives_2d + self.max_relative_position #  converted from distance matrix (-ve & +ve) to a positive distance matrix represented as tokens for each position
        # Make in type long
        clamped_relatives_positive = torch.LongTensor(clamped_relatives_positive).to(self.device) # Q: should we make .device from here? Q: why to make it in "Long"?
        # this is the matrix that we want to input for embeddings to be learnt (each position will be represented by 512 embedding vector)
        # get the relative position embeddings [embedding vector for each relative position token]
        relative_position_embeddings = self.embedding_table(clamped_relatives_positive)
        # shape: len_q x len_k x head_dim
        return relative_position_embeddings

class RelativePositionalEmbedding(nn.Module):
    def __init__(self, emb_dim, n_heads, max_relative_position, dropout, device): # here no need of vocab size as we can already infer from max_relative_pos
        super().__init__() # Q: do we need to add device information here?
        self.dropout, self.max_relative_position = dropout, max_relative_position
        # N_heads: instead of having one big embedding vector (512) we make 2 (256) --> both will pass on the same input and same everything
        self.emb_dim, self.n_heads = emb_dim, n_heads
        self.head_dim = emb_dim // n_heads

        # layers
        # This takes embedding and output embedding all of same dimension 
        # --> reduction of head_dim happens during operations of attention but at last concatenated and pass through the linear
        self.q_linear, self.k_linear, self.v_linear = nn.Linear(emb_dim, emb_dim), nn.Linear(emb_dim, emb_dim), nn.Linear(emb_dim, emb_dim) # 
        self.out = nn.Linear(emb_dim, emb_dim) # the output is emb_dim since we still want to o.p emb as this is pos emb (not vocab)
        self.relative_position_k = RelativePosition(max_relative_position, self.head_dim, device)
        self.relative_position_v = RelativePosition(max_relative_position, self.head_dim, device)
        self.scale_factor = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, atten_mask=None): # Q: mask here is the attention mask, right?
        # --------------------------------- Attention QK ----------------------- Question: what is the output from this? similarity between each word with all other words?
        # Beginning of Equation (5) --> e_ij
        # 1. normal self attention (qk) --> [ x*WQ * (x*WK)^T ]
        # q.shape: [batch_size, seq_len, emb_dim]
        batch_size, len_q, len_k, len_v = q.shape[0], q.shape[1], k.shape[1], v.shape[1]
        q, k = self.q_linear(q), self.k_linear(k)
        # QA: what is the difference between view and permute --> view: split/ chunk/ combine (reads data sequentially in any way you want), permute: only transpose what's already there
        q_heads = q.view(batch_size, -1 ,self.n_heads, self.head_dim) 
        k_heads = k.view(batch_size, -1 ,self.n_heads, self.head_dim) 
        # for each batch: we have each sequence defined by n heads, each head has head dim (10 sequences, each seq has 2 heads, each head has 256 dim)
        # what we want is to have, n heads: each head has n sequences and each sequence is defined by head_dim (2 heads, each head has 10 sequences; each sequence has 256 emb vector to define it)
        # We do this to multiply the needed parts together which are the emb dim and seq_len; as matrix multiple only happen to the last 2 indices
        q_head_perm, k_head_perm = q_heads.permute(0, 2, 1, 3), k_heads.permute(0, 2, 1, 3) # for we want to calculate emb_dim * seq (this is what matters)
        # q: [1, 2, 10, 256] -- k: [1, 2, 10, 256] --> k.T: [1, 2, 256, 10]
        qk = torch.matmul(q_head_perm, k_head_perm.permute(0, 1, 3, 2)) 
        # shape: batch_size, n_heads, len_q, len_k

        # ----------------------- Relatives ------------------------
        # 2. Relative of k --> [a_k]
        # [batch_size, len_sequence, emb_dim]
        r_k = self.relative_position_k(len_q, len_k)
        # shape: len_q x len_k x head_dim --> r_k.T: len_q x head_dim x len_k
        q2_r = q_heads.permute(1, 0, 2, 3).contiguous().view(len_q, batch_size*self.n_heads, self.head_dim)
        # shape: len_q x bsz*n_head x head_dim 
        q_rk = torch.matmul(q2_r, r_k.transpose(1, 2)) # transpose only swaps the two indices together
        # shape: len_q x bsz*n_head x len_k --> we want to make bsz and n_head at first and leave interesting points to last 2 indices
        q_rk = q_rk.transpose(0, 1)
        # shape: bsz*n_head x len_q x len_k 
        q_rk = q_rk.contiguous().view(batch_size, self.n_heads, len_q, len_k)
        # shape: batch_size, n_heads, len_q, len_k
        attn1, attn2 = qk, q_rk
        attn_total = (attn1 + attn2) / self.scale_factor
        # shape: batch_size, n_heads, len_q, len_k
        #if atten_mask is not None: # Question: ask hazem on how to adjust for mask
        #   attn_total = attn_total.masked_fill(atten_mask == 0, -1e10)
        
        # End of Equation (5)
        # ------------------------ Value --------------------------

        # Begining of Equation (3)
        # 1. Softmax of total pre attention: alpha_ij = softmax (e_ij)
        attn_soft = self.dropout(F.softmax(attn_total, dim=-1))
        # shape: batch_size, n_heads, len_q, len_k

        # 3. Linear v -->  x*W_v
        v = self.v_linear(v)
        # shape: batch_size, seq_len, emb_dim
        v_heads = v.view(batch_size, -1, self.n_heads, self.head_dim) # multiply last from 1st with prelast from 2nd
        v_heads = v_heads.permute(0, 2, 1, 3)
        # shape: batch_size, n_heads, seq_len, head_dim
        # 4. Softmax * linear v --> alpha_ij * [x*W_v]
        # For matrix mult: they should be same exact sizes except for last two + last_first == prelast_sec
        weight1 = torch.matmul(attn_soft, v_heads)
        # shape: batch_size, n_heads, len_q, head_dim

        # 2. Relative position of v --> a_v
        r_v = self.relative_position_v(len_q, len_v) 
        # shape: len_q, len_v, head_dim
        attn_soft = attn_soft.permute(2, 0, 1, 3).contiguous().view(len_q, batch_size*self.n_heads, len_k)
        # shape: len_q, bsz*n_heads, len_k ## Question: how is len_v same as len_k? (for correct multiplication -- how do we know)
        # 5. Softmax * relative v --> alpha_ij * [a_v]
        weight2 = torch.matmul(attn_soft, r_v)
        # shape: len_q, bsz*n_heads, head_dim
        weight2 = weight2.transpose(0, 1).contiguous().view(batch_size, self.n_heads, len_q, self.head_dim)
        # shape: batch_size, n_heads, len_q, head_dim
        # 6. summation of (4) & (5)
        weights = weight1 + weight2
        # shape: batch_size, n_heads, len_q, head_dim
        weights = weights.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.emb_dim)
        # shape: batch_size, len_q, emb_dim

        ### linear over summation ###
        out = self.out(weights) 
        # needed out_shape: batch_size, len_q, emb_dim
        return out
        


        







# class SelfAttention(nn.Module):
#     def __init__(self, emb_size, clip, seq_len):
#         self.emb_size = emb_size
#         self.q_linear = nn.Linear(emb_size, emb_size)
#         self.k_linear = nn.Linear(emb_size, emb_size)
#         self.v_linear = nn.Linear(emb_size, emb_size)
#         self.max_clip = max(-clip, min(seq_len, clip))
        
#         clip = self.clip

#     def forward(self, q, k, v, mask=None): # which one is the attention weights? and which is q/k/v weights?
#         q, k, v = self.q_linear(q), self.k_linear(k), self.v_linear(v)

#         # eq(5.1) == eq(2) --> qk_scale
#         qk_scale = (torch.matmul(q, k.transpose(-2, -1)))/math.sqrt(self.emb_size)
#         # eq(5.2)

#         k_relative = k[] # k[-] -- put clip?
#         qk_relative = (torch.matmul(q, k_relative.transpose(-1, -2)))/math.sqrt(self.emb_size)

#         qk_total = qk_scale + qk_relative
#         qk_total_soft = F.softmax(qk_total, dim=-1)
        
#         # eq(3)
#         v_relative = v[]
#         v_new = v + v_relative
        
        
#         qkv = torch.matmul(qk_total_soft, v_new)
