from torch import nn
import torch

class RelativePosition(nn.Module):

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Embedding(max_relative_position * 2 + 1, num_units) # max_relative from right and left (*2) + 1 (for the relative with myself)
        # nn.init.xavier_normal_(self.embeddings_table)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position) # clip all indices to -ve and +ve of max relative position
        final_mat = distance_mat_clipped + self.max_relative_position #
        final_mat = torch.LongTensor(final_mat).cuda()
        embeddings = self.embeddings_table(final_mat).cuda() # 

        return embeddings

class RelativePositionalEmbedding(nn.Module):
    def __init__(self, hid_dim, n_heads, max_relative_position, dropout, device):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.max_relative_position = max_relative_position

        self.relative_position_k = RelativePosition( self.head_dim, self.max_relative_position,) # k_relative
        self.relative_position_v = RelativePosition(  self.head_dim, self.max_relative_position) #  Q: why two different? why not both same initialization? 

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask = None):
        #query = [batch size, query len, hid dim] ## Question: how is this key/query/value with hidden dim? does this mean we input: token embedding? q/ k/ v
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
        batch_size = query.shape[0]
        len_k = key.shape[1]
        len_q = query.shape[1] # Q: isn't this the sequence length (or maybe part of sentence?)
        len_v = value.shape[1]

        query = self.fc_q(query)
        key = self.fc_k(key)
        value = self.fc_v(value)

        r_q1 = query.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        r_k1 = key.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        attn1 = torch.matmul(r_q1, r_k1.permute(0, 1, 3, 2)) # q: is the attention qk only? then what does happen when v is there too? (projection?)

# --------------------Set  1--------------------
        r_k2 = self.relative_position_k(len_q, len_k) # seq_len * seq_len * head_dim [q_len, k_len, num_units]
        # Q1: is query.view change in the view of query or just its output `r_q1`
        r_q2 = query.permute(1, 0, 2).contiguous().view(len_q, batch_size*self.n_heads, self.head_dim) # Q: why combine them? and how do we multiply batch with n_heads --> multiplication to be able to make matrix multiplication
        # r_k2: is the only relative
        attn2 = torch.matmul(r_q2, r_k2.transpose(1, 2)).transpose(0, 1)
        attn2 = attn2.contiguous().view(batch_size, self.n_heads, len_q, len_k)
        attn = (attn1 + attn2) / self.scale

        
        # if mask is not None:
        #   attn = attn.masked_fill(mask == 0, -1e10)

# ----------------Set  2-------------------------------

        attn = self.dropout(torch.softmax(attn, dim = -1))
        #attn = [batch size, n heads, query len, key len] 


        r_v1 = value.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        weight1 = torch.matmul(attn, r_v1)

        r_v2 = self.relative_position_v(len_q, len_v) 
        weight2 = attn.permute(2, 0, 1, 3).contiguous().view(len_q, batch_size*self.n_heads, len_k) # Q: how is weight2 from the original attn? what part of equation is this
        weight2 = torch.matmul(weight2, r_v2)
        
        weight2 = weight2.transpose(0, 1).contiguous().view(batch_size, self.n_heads, len_q, self.head_dim)

        x = weight1 + weight2
        
        #x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        
        #x = [batch size, query len, hid dim]
        
        x = self.fc_o(x)
        
        #x = [batch size, query len, hid dim]
        
        return x