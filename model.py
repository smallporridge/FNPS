""" Construct the model """
import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from Layers import EncoderLayer, DecoderLayer
cudaid=0
def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(0).type(torch.float).unsqueeze(-1)

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table according to sin/cos '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''
    len_q = seq_q.size(1)
    # padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = seq_k.eq(0)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

class Encoder(nn.Module):
    """ A transformer encoder layer 
        len_max_seq: max length of the input sequence
        embed_dim: embed dimension of the input
        d_model: dimension of input, in the first layer, equal to embed_dim
        d_inner: 
        n_layers: 
    """
    def __init__(self, len_max_seq, embed_dim, d_model, d_inner, n_layers, n_head,
        d_k, d_v, dropout=0.1):
        super().__init__()
        n_position = len_max_seq + 1

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, embed_dim, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_emb, src_pos, atten_mask=None, return_attns=False, needpos=False):
        enc_slf_attn_list = []

        # -- Prepare mask
        # print("atten_mask: ", atten_mask)
        if atten_mask == None:
            slf_attn_mask = get_attn_key_pad_mask(seq_k=src_pos, seq_q=src_pos)
        else:
            slf_attn_mask = atten_mask
        
        non_pad_mask = get_non_pad_mask(src_pos)

        # -- Forward
        if needpos:
            enc_output = src_emb + self.position_enc(src_pos)
        else:
            enc_output = src_emb

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,

class Dynamic_Attention(nn.Module):
	'''
		attention layer
	'''
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)
        self.wq = nn.Linear(100, 256)
        self.wk = nn.Linear(100, 256)
        self.att = nn.Linear(256, 1)

    def forward(self, q, k, pos, needmask=False):
        query = q.unsqueeze(1).expand(-1,k.size(1),-1)
        query, key= self.wq(query), self.wk(k)
        attn = self.att(query + key).squeeze(2)
        if needmask:
            mask = pos.eq(0)
            attn = attn.masked_fill(mask, -np.inf)
        attn = self.softmax(attn).unsqueeze(2)
        #attn = self.dropout(attn)
        output = torch.sum(attn*k, 1)
        #output = torch.bmm(attn, v)

        return output

class Contextual(nn.Module):
	'''
		The model FNPS
		query: current query
		doc1: positive document
		doc2: negative document
		features1: additional features of doc1
		features2: additional features of doc2
		delta: parameter of lambdarank loss
		label: 0,1
		long_history: user's long-term history
		short_history: user's short-term history
		short_pos: short-term history position, [1,2,3,...,0,0,0]
		long_pos: long-term history position, [1,2,3,...,0,0,0]
		lfriend_log: behavior-based friends' query log
		lfriend_pos: behavior-based friends' query log position
		lfriend_pos_mask: behavior-based friends' padding
		lfriend_att_mask: behavior-based friends' adjacent matrix
		sfriend_log: relation-based friends' query log
		sfriend_pos: relation-based friends' query log position
		sfriend_pos_mask: relation-based friends' padding
		sfriend_att_mask: relation-based friends' adjacent matrix
		cross_att_mask: adjacent matrix of cross attention
	'''
    def __init__(self, max_friendnum, max_lcircle_num, max_scircle_num, max_friendnum, max_sess_len, queryhis_len, embed_dim, batch_size, embed_path, 
        vocab_path, d_model=100, d_inner=512, n_layers=1, n_head=8, d_k=64, d_v=64, dropout=0.1):

        super().__init__()

        self.max_friendnum = max_friendnum # max number of friends in each circle
        self.max_lcircle_num = max_lcircle_num # max number of behavior-based friend circle
        self.max_scircle_num = max_scircle_num # max number of relation-based friend circle
        self.max_sess_len = max_sess_len # max length of short-term history
        self.queryhis_len = queryhis_len # max length of long-term history
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.friend_attention = Dynamic_Attention()
        self.personal_attention = Dynamic_Attention()

        self.encoder_friends = Encoder(len_max_seq=friendnum+1, embed_dim=embed_dim, d_model=d_model, # GAT, masked transformer
            d_inner=d_inner, n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout)

        self.encoder_session = Encoder(len_max_seq=max_sess_len, embed_dim=embed_dim, d_model=d_model, # short-term transformer
            d_inner=d_inner, n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout)

        self.encoder_history = Encoder(len_max_seq=queryhis_len, embed_dim=embed_dim, d_model=d_model, # long-term transformer
            d_inner=d_inner, n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout)

        self.cross_attention = Encoder(len_max_seq=max_lcircle_num+max_scircle_num, embed_dim=embed_dim, d_model=d_model, # cross-attention layer
            d_inner=d_inner, n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout)

        #self.feature_layer = nn.Sequential(nn.Linear(110, 1),nn.Tanh())
        self.score_layer = nn.Sequential(nn.Linear(125, 1),nn.Tanh())
        self.criterion = nn.CrossEntropyLoss()

    def pairwise_loss(self, score1, score2):
        return (1/(1+torch.exp(score2-score1)))

    def forward(self, query, doc1, doc2, feature1, feature2, delta, label, long_history, long_pos, short_history, short_pos, lfriend_log, lfriend_pos, lfriend_pos_mask, lfriend_att_mask, sfriend_log, sfriend_pos, sfriend_pos_mask, sfriend_att_mask, cross_att_mask):
        lfriend_log = lfriend_log.view(-1, self.queryhis_len, self.embed_dim)
        lfriend_pos = lfriend_pos.view(-1, self.queryhis_len)
        lfriend_log, *_ = self.encoder_history(lfriend_log, lfriend_pos)
        lfriend_log = torch.mean(lfriend_log, 1).view(-1, self.friendnum+1, self.embed_dim)
        lfriend_pos_mask = lfriend_pos_mask.view(-1, self.friendnum+1)
        lfriend_log, *_ = self.encoder_friends(lfriend_log, lfriend_pos_mask, atten_mask=lfriend_att_mask)
        lfriend_log = lfriend_log[:,-1:,:].view(-1, max_lcircle_num, self.embed_dim)
        
        sfriend_log = sfriend_log.view(-1, self.queryhis_len, self.embed_dim)
        sfriend_pos = sfriend_pos.view(-1, self.queryhis_len)
        sfriend_log, *_ = self.encoder_history(sfriend_log, sfriend_pos, needpos=True)
        sfriend_log = torch.mean(sfriend_log, 1).view(-1, self.friendnum+1, self.embed_dim)
        sfriend_pos_mask = sfriend_pos_mask.view(-1, self.friendnum+1)
        sfriend_log, *_ = self.encoder_friends(sfriend_log, sfriend_pos_mask, atten_mask=sfriend_att_mask)
        sfriend_log = sfriend_log[:,-1:,:].view(-1, max_scircle_num, self.embed_dim)

        friend_log = torch.cat([lfriend_log, sfriend_log], 1)
        friend_pos = torch.ones(friend_log.size(0), friend_log.size(1)).cuda(cudaid)
        friend_log, *_ = self.cross_attention(friend_log, friend_pos, atten_mask=cross_att_mask)

        short_history = torch.cat([short_history, query.unsqueeze(1)], 1)
        long_history, *_ = self.encoder_history(long_history, long_pos, needpos=True)
        short_history, *_ = self.encoder_session(short_history, short_pos, needpos=True)
        q_encode = short_history[:,-1:,:].squeeze()

        friend_output = self.friend_attention(q_encode, friend_log, friend_pos)
        personal_output = self.personal_attention(q_encode, long_history, long_pos, needmask=True)

        # compute matching scores
        score_pos_qs = torch.cosine_similarity(q_encode, doc1, dim=1).unsqueeze(1)
        score_neg_qs = torch.cosine_similarity(q_encode, doc2, dim=1).unsqueeze(1)

        score_pos_friend = torch.cosine_similarity(friend_output, doc1, dim=1).unsqueeze(1)
        score_neg_friend = torch.cosine_similarity(friend_output, doc2, dim=1).unsqueeze(1)

        score_pos_personal = torch.cosine_similarity(personal_output, doc1, dim=1).unsqueeze(1)
        score_neg_personal = torch.cosine_similarity(personal_output, doc2, dim=1).unsqueeze(1)

        # score_pos_q = torch.cosine_similarity(query, doc1, dim=1).unsqueeze(1)
        # score_neg_q = torch.cosine_similarity(query, doc2, dim=1).unsqueeze(1)

        # score_pos_feature = self.feature_layer(feature1)
        # score_neg_feature = self.feature_layer(feature2)

        score_pos = torch.cat([score_pos_qs, score_pos_friend, score_pos_personal, feature1], 1)
        score_1 = self.score_layer(score_pos)
        score_neg = torch.cat([score_neg_qs, score_neg_friend, score_neg_personal, feature2], 1)
        score_2 = self.score_layer(score_neg)

        score = torch.cat([score_1, score_2], 1)

        p_score = torch.cat([self.pairwise_loss(score_1, score_2),
                    self.pairwise_loss(score_2, score_1)], 1)

        preds = F.softmax(score, 1)

        loss = self.criterion(p_score, label)

        return score, preds, loss