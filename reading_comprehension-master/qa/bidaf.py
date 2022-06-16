import torch
from torch import nn, Tensor, tensor
import torch.nn.functional as F
from pytorch_pretrained_bert import BertModel
from pytorch_pretrained_bert.modeling import BertPreTrainedModel
from . import load_data
from pathlib import Path
from typing import List, Any
from dataclasses import dataclass
from .load_data import Example
from pytorch_pretrained_bert.tokenization import BertTokenizer
import dataclasses

class Similarity(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.wq = nn.Linear(d, 1, bias=False)
        self.wc = nn.Linear(d, 1, bias=False)
        self.wqc = nn.Linear(d, 1, bias=False)

    def forward(self, context: Tensor, context_mask, query: Tensor, query_mask):
        Q = query.size(1)
        x_q = self.wq(query).squeeze(-1)
        x_c = self.wc(context).squeeze(-1)
        x_list = []
        for q in range(Q):
            x = query[:, q, :]
            x = x.unsqueeze(1)
            x_qc = self.wqc(x * context)
            x_list.append(x_qc)
        x_qc = torch.cat(x_list, 2)
        sim = x_q.unsqueeze(1) + x_c.unsqueeze(2) + x_qc
        inf = context.new_tensor(float('-inf'))
        pos_inf = context.new_tensor(float('inf'))
        one = context.new_tensor(1.0)
        sim = sim * torch.where(context_mask==1, one, inf).unsqueeze(2)
        sim = sim * torch.where(query_mask==1, one, inf).unsqueeze(1)
        # For the case where both the context and query mask are 0
        sim = torch.where(sim==pos_inf, inf, sim)
        return sim


class Query2Context(nn.Module):
    def forward(self, context: Tensor, sim: Tensor):
        sim = sim.max(dim=2)[0]
        sim = F.softmax(sim, dim=1)
        context_attended = torch.einsum('bc, bcd->bd', [sim, context])
        return context_attended


class Context2Query(nn.Module):
    def forward(self, query: Tensor, sim: Tensor):
        sim = F.softmax(sim, dim=2)
        context_attended = torch.einsum('bcq, bqd->bcd', [sim, query])
        return context_attended


class CombineAttention(nn.Module):
    def forward(self, context: Tensor, context2query: Tensor, query2context: Tensor):
        query2context = query2context.unsqueeze(1)
        context_context2query = context * context2query
        context_query2context = context * query2context
        res = torch.cat([context, context2query, context_context2query, context_query2context], dim=2)
        return res


class Attention(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.similarity = Similarity(d)
        self.q2c = Query2Context()
        self.c2q = Context2Query()
        self.combine = CombineAttention()

    def forward(self, context: Tensor, context_mask, query: Tensor, query_mask):
        sim = self.similarity(context, context_mask, query, query_mask)
        q2c:Tensor = self.q2c(context, sim)
        c2q:Tensor = self.c2q(query, sim)
        g = self.combine.forward(context, c2q, q2c)
        return g


class Modeling(nn.Module):
    def __init__(self, d: int, h: int):
        super().__init__()
        self.rnn = nn.LSTM(d, h, num_layers=2, bidirectional=True)

    def forward(self, g: Tensor):
        x = self.rnn.forward(g)[0]
        return x


class Bidaf(nn.Module):
    def __init__(self, d: int, h: int):
        super().__init__()
        self.attention = Attention(d)
        self.modeling = Modeling(4 * d, h)

    def forward(self, context: Tensor, context_mask,query: Tensor,query_mask):
        g:Tensor = self.attention(context, context_mask,query,query_mask)
        x:Tensor = self.modeling(g)
        return x

class BidafQA(nn.Module):
    def __init__(self, d: int, h: int):
        super().__init__()
        self.bidaf = Bidaf(d, h)
        self.classifier = nn.Linear(2*h, 1)

    def forward(self, context: Tensor, context_mask, query: Tensor, query_mask):
        batch_size, num_choices, n_context, d = context.size()
        n_query = query.size(2)
        context = context.view(-1, n_context, d)
        query = query.view(-1, n_query, d)
        x:Tensor = self.bidaf(context, context_mask, query, query_mask)
        x = x[:, 0, :]
        logits = self.classifier(x)
        logits = logits.view(batch_size, num_choices)
        return logits

D = 768

class BertEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, *args):
        return self.bert(*args, output_all_encoded_layers=False)[0]

class BidafBert(nn.Module):
    def __init__(self, h:int):
        super().__init__()
        self.embedding = BertEmbedding()
        self.qa = BidafQA(D, h)

    def forward(self, context: Tensor, context_mask, context_seq, query: Tensor, query_mask, query_seq, labels=None):
        batch_size, num_choices, n_context = context.size()
        batch_size, num_choices, n_query = query.size()
        context = context.view(-1, n_context)
        context_seq = context_seq.view(-1, n_context)
        context_mask = context_mask.view(-1, n_context)
        context = self.embedding(context, context_seq, context_mask)
        context = context.view(batch_size, num_choices, n_context, D)
        query = query.view(-1, n_query)
        query_seq = query_seq.view(-1, n_query)
        query_mask = query_mask.view(-1, n_query)
        query = self.embedding(query, query_seq, query_mask).view(batch_size, num_choices, n_query, D)
        logits = self.qa(context, context_mask, query, query_mask)
        if labels is None:
            return logits
        else:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return loss



def read_training()->List[Example]:
    return torch.load(Path(__file__).parent.parent/'data/RACE/race.pickle')

def read_test()->List[Example]:
    return torch.load(Path(__file__).parent.parent/'data/RACE/race_test.pickle')


def tokenize(sentence: str, max_len:int, tokenizer):
    tokens:List[str] = tokenizer.tokenize(sentence)
    tokens.insert(0, "[CLS]")
    tokens = tokens[:max_len]
    ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(ids)
    segment_ids = [0] * len(ids)
    padding = [0] * (max_len - len(ids))
    ids += padding
    input_mask += padding
    segment_ids += padding
    return ids, input_mask, segment_ids

@dataclass
class Features:
    context: Tensor
    context_mask: Tensor
    context_seq: Tensor
    query: Tensor
    query_mask: Tensor
    query_seq: Tensor
    label: Tensor

def feature_subset(f:Features, N: int):
    d = dataclasses.asdict(f)
    new_k = {}
    for k, v in d.items():
        new_k[k] = v[:N, ...]
    return Features(**new_k)

def tokenize_all(examples: List[Example], max_context_len:int=400, max_query_len:int=50):
    tokenizer = get_tokenizer()
    n_examples = len(examples)
    # n_examples = 10
    context_out = torch.empty([n_examples, 4, max_context_len], dtype=torch.long)
    context_mask_out = torch.empty([n_examples, 4, max_context_len], dtype=torch.long)
    context_seq_out = torch.empty([n_examples, 4, max_context_len], dtype=torch.long)
    query_out = torch.empty([n_examples, 4, max_query_len], dtype=torch.long)
    query_mask_out = torch.empty([n_examples, 4, max_query_len], dtype=torch.long)
    query_seq_out = torch.empty([n_examples, 4, max_query_len], dtype=torch.long)
    labels_out = torch.empty(n_examples, dtype=torch.long)
    for example_idx, example in enumerate(examples):
        if example_idx >= n_examples:
            break
        context, context_mask, context_seq = tokenize(example.context_sentence, max_context_len, tokenizer)
        for ending_idx, ending in enumerate(example.endings):
            query_str = example.start_ending + " " + ending
            query, query_mask, query_seq = tokenize(query_str, max_query_len, tokenizer)
            context_out[example_idx, ending_idx, :] = torch.tensor(context)
            context_mask_out[example_idx, ending_idx, :] = torch.tensor(context_mask)
            context_seq_out[example_idx, ending_idx, :] = torch.tensor(context_seq)
            query_out[example_idx, ending_idx, :] = torch.tensor(query)
            query_mask_out[example_idx, ending_idx, :] = torch.tensor(query_mask)
            query_seq_out[example_idx, ending_idx, :] = torch.tensor(query_seq)
        labels_out[example_idx] = example.label
    return Features(context_out, context_mask_out, context_seq_out, query_out, query_mask_out, query_seq_out, labels_out)

def save_features():
    examples = read_training()
    features = tokenize_all(examples)
    torch.save(features, Path(__file__).parent.parent/'data/RACE/features.pickle')

    examples = read_test()
    features = tokenize_all(examples)
    torch.save(features, Path(__file__).parent.parent/'data/RACE/features_test.pickle')

def get_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    return tokenizer


