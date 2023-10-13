# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn as nn
#from scipy.optimize import linear_sum_assignment
import numpy as np

def linear_sum_assignment_new(cost_matrix):
    cost_matrix = np.array(cost_matrix)
    n, m = cost_matrix.shape

    if n > m:
        raise ValueError("The cost matrix must have more rows than columns")

    # Step 1: Subtract the minimum value in each row from that row
    cost_matrix -= cost_matrix.min(axis=1, keepdims=True)

    # Step 2: Subtract the minimum value in each column from that column
    cost_matrix -= cost_matrix.min(axis=0, keepdims=True)

    # Step 3: Cover all zeros with the minimum number of lines
    mask = np.zeros_like(cost_matrix, dtype=bool)
    cover_rows, cover_cols = np.zeros(n, dtype=bool), np.zeros(m, dtype=bool)
    while not np.all(cover_rows) or not np.all(cover_cols):
        mask.fill(False)
        mask[~cover_rows, :] = True
        mask[:, ~cover_cols] = True
        marked_zeros = np.zeros_like(cost_matrix, dtype=bool)
        marked_zeros[mask] = cost_matrix[mask] == 0

        # Step 4: Find a non-covered zero and prime it
        row_has_marked_zero = np.any(marked_zeros, axis=1)
        if not np.any(row_has_marked_zero):
            # Step 6: Find the smallest uncovered value
            min_uncovered_value = np.min(cost_matrix[~cover_rows, ~cover_cols])
            cost_matrix[~cover_rows, ~cover_cols] -= min_uncovered_value
            min_uncovered_value_mask = cost_matrix[~cover_rows, ~cover_cols] == 0
            cost_matrix[~cover_rows, ~cover_cols][min_uncovered_value_mask] = np.inf
        else:
            row = np.where(row_has_marked_zero)[0][0]
            col = np.where(marked_zeros[row, ~cover_cols])[0][0]
            cover_cols[col] = True
            cover_rows[row] = False

    # Step 5: Find a covered zero and prime it, then
    #           uncover the row containing the primed zero and cover the column containing the starred zero
    solution = np.full(n, -1, dtype=int)
    for row in range(n):
        col = np.where(mask[row, :])[0][0]
        solution[row] = col

    return solution, cost_matrix[solution != -1].sum()



def span_similarity(span1, span2):
    # Define a similarity score between two spans.
    # You can use various metrics such as Jaccard similarity, overlap ratio, etc.
    # For example, you can calculate Jaccard similarity:
    intersection = len(set(span1) & set(span2))
    union = len(set(span1) | set(span2))
    return intersection / union if union > 0 else 0
def calculate_similarity_matrix(test_list, gold_list):
    # Calculate a similarity matrix where each cell (i, j) represents
    # the similarity score between test_list[i] and gold_list[j].
    similarity_matrix = np.zeros((len(test_list), len(gold_list)))
    for i, test_span in enumerate(test_list):
        for j, gold_span in enumerate(gold_list):
            similarity_matrix[i][j] = span_similarity(test_span, gold_span)
    return similarity_matrix

def find_best_matching(test_list, gold_list):
    # Example usage with different lengths:
    # test_list = [(0, 5), (10, 15), (20, 25)]
    # gold_list = [(2, 7), (11, 16)]

    # best_matching, match_rate = find_best_matching(test_list, gold_list)
    # print("Best Matching:", best_matching)
    # print("Overall Match Rate:", match_rate)


    # Calculate the similarity matrix.
    similarity_matrix = calculate_similarity_matrix(test_list, gold_list)
    
    # Use the Hungarian algorithm to find the optimal assignment that maximizes the overall match rate.
    row_indices, col_indices = linear_sum_assignment_new(-similarity_matrix)
    
    # Calculate the overall match rate.
    if len(test_list)>0:
        match_rate = similarity_matrix[row_indices, col_indices].sum() / len(test_list)
    else:
        match_rate = 0
    # Create a dictionary to represent the best matching.
    best_matching = {}
    for i, j in zip(row_indices, col_indices):
        if i < len(test_list) and j < len(gold_list):
            best_matching[test_list[i]] = gold_list[j]
    
    return best_matching, match_rate
class MetricsCalculator(object):
    def __init__(self):
        super().__init__()

    def get_sample_f1(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()
        return 2 * torch.sum(y_true * y_pred) / torch.sum(y_true + y_pred)

    def get_sample_precision(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()
        return torch.sum(y_pred[y_true == 1]) / (y_pred.sum() + 1)

    def get_evaluate_fpr(self, y_pred, y_true):
        X, Y, Z = 1e-10, 1e-10, 1e-10
        y_pred = y_pred.data.cpu().numpy()
        y_true = y_true.data.cpu().numpy()
        pred = []
        true = []
        for b, l, start, end in zip(*np.where(y_pred > 0)):
            pred.append((b, l, start, end))
        for b, l, start, end in zip(*np.where(y_true > 0)):
            true.append((b, l, start, end))
        R = set(pred)
        T = set(true)
        X = len(R & T)
        Y = len(R)
        Z = len(T)
        if Y==0 or Z==0:
            return 0,0,0
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        return f1, precision, recall

    def get_evaluate_iou(self, y_pred, y_true):
        X_total, Y_total, Z_total = 1e-10, 1e-10, 1e-10
        y_pred = y_pred.data.cpu().numpy()
        y_true = y_true.data.cpu().numpy()

        fpr_dict = {}  # Dictionary to store FPR for each (b, l) combination

        for b in range(y_pred.shape[0]):
            for l in range(y_pred.shape[1]):
                pred = y_pred[b, l, :, :]
                true = y_true[b, l, :, :]

                pred_positive = pred > 0
                true_positive = true > 0

                # gold_list: [(span_text,span)]
                correct_rate = 0
                total = 0

                try:
                    best_match,match_rate = find_best_matching(pred_positive,true_positive)
                except:
                    print(pred_positive)
                    print(true_positive)
                correct_rate+=match_rate*len(best_match)
                total+=len(best_match)
        if total!=0:
            accuracy = correct_rate/total
        else :
            accuracy = -1
        return accuracy
class RawGlobalPointer(nn.Module):
    def __init__(self, encoder, ent_type_size, inner_dim, RoPE=True):
        # encodr: RoBerta-Large as encoder
        # inner_dim: 64
        # ent_type_size: ent_cls_num
        super().__init__()
        self.encoder = encoder
        self.ent_type_size = ent_type_size
        self.inner_dim = inner_dim
        self.hidden_size = encoder.config.hidden_size
        self.dense = nn.Linear(self.hidden_size, self.ent_type_size * self.inner_dim * 2)

        self.RoPE = RoPE

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(self.device)
        return embeddings

    def forward(self, input_ids, attention_mask, token_type_ids):
        self.device = input_ids.device

        context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        # last_hidden_state:(batch_size, seq_len, hidden_size)
        last_hidden_state = context_outputs[0]

        batch_size = last_hidden_state.size()[0]
        seq_len = last_hidden_state.size()[1]

        outputs = self.dense(last_hidden_state)
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        outputs = torch.stack(outputs, dim=-2)
        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]
        if self.RoPE:
            # pos_emb:(batch_size, seq_len, inner_dim)
            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim)
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
        # logits:(batch_size, ent_type_size, seq_len, seq_len)
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)

        # padding mask
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        logits = logits * pad_mask - (1 - pad_mask) * 1e12

        # 排除下三角
        mask = torch.tril(torch.ones_like(logits), -1)
        logits = logits - mask * 1e12

        return logits / self.inner_dim ** 0.5

class SinusoidalPositionEmbedding(nn.Module):
    """定义Sin-Cos位置Embedding
    """
    def __init__(
        self, output_dim, merge_mode='add', custom_position_ids=False):
        super(SinusoidalPositionEmbedding, self).__init__()
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.custom_position_ids = custom_position_ids

    def forward(self, inputs):
        if self.custom_position_ids:
            seq_len = inputs.shape[1]
            inputs,position_ids = inputs
            position_ids = position_ids.type(torch.float)
        else:
            input_shape = inputs.shape
            batch_size, seq_len = input_shape[0], input_shape[1]
            position_ids = torch.arange(seq_len).type(torch.float)[None]
        indices = torch.arange(self.output_dim // 2).type(torch.float)
        indices = torch.pow(10000.0, -2 * indices / self.output_dim)
        embeddings = torch.einsum('bn,d->bnd', position_ids, indices)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = torch.reshape(embeddings, (-1, seq_len, self.output_dim))
        if self.merge_mode == 'add':
            return inputs + embeddings.to(inputs.device)
        elif self.merge_mode == 'mul':
            return inputs * (embeddings + 1.0).to(inputs.device)
        elif self.merge_mode == 'zero':
            return embeddings.to(inputs.device)

class EffiGlobalPointer(nn.Module):
    def __init__(self, encoder, ent_type_size, inner_dim, RoPE=True):
        #encodr: RoBerta-Large as encoder
        #inner_dim: 64
        #ent_type_size: ent_cls_num
        super(EffiGlobalPointer, self).__init__()
        self.encoder = encoder
        self.ent_type_size = ent_type_size
        self.inner_dim = inner_dim
        self.hidden_size = encoder.config.hidden_size
        self.RoPE = RoPE

        self.dense_1 = nn.Linear(self.hidden_size, self.inner_dim * 2)
        self.dense_2 = nn.Linear(self.hidden_size, self.ent_type_size * 2) #原版的dense2是(inner_dim * 2, ent_type_size * 2)

    def sequence_masking(self, x, mask, value='-inf', axis=None):
        if mask is None:
            return x
        else:
            if value == '-inf':
                value = -1e12
            elif value == 'inf':
                value = 1e12
            assert axis > 0, 'axis must be greater than 0'
            for _ in range(axis - 1):
                mask = torch.unsqueeze(mask, 1)
            for _ in range(x.ndim - mask.ndim):
                mask = torch.unsqueeze(mask, mask.ndim)
            return x * mask + value * (1 - mask)

    def add_mask_tril(self, logits, mask):
        if mask.dtype != logits.dtype:
            mask = mask.type(logits.dtype)
        logits = self.sequence_masking(logits, mask, '-inf', logits.ndim - 2)
        logits = self.sequence_masking(logits, mask, '-inf', logits.ndim - 1)
        # 排除下三角
        mask = torch.tril(torch.ones_like(logits), diagonal=-1)
        logits = logits - mask * 1e12
        return logits

    def forward(self, input_ids, attention_mask, token_type_ids):
        self.device = input_ids.device
        
        context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        last_hidden_state = context_outputs.last_hidden_state
        outputs = self.dense_1(last_hidden_state)
        qw, kw = outputs[...,::2], outputs[..., 1::2] #从0,1开始间隔为2
        if self.RoPE:
            pos = SinusoidalPositionEmbedding(self.inner_dim, 'zero')(outputs)
            cos_pos = pos[..., 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos[..., ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], 3)
            qw2 = torch.reshape(qw2, qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], 3)
            kw2 = torch.reshape(kw2, kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
        logits = torch.einsum('bmd,bnd->bmn', qw, kw) / self.inner_dim**0.5
        bias = torch.einsum('bnh->bhn', self.dense_2(last_hidden_state)) / 2
        logits = logits[:, None] + bias[:, ::2, None] + bias[:, 1::2, :, None] #logits[:, None] 增加一个维度
        logits = self.add_mask_tril(logits, mask=attention_mask)
        return logits
