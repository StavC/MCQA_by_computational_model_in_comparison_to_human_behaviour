from pathlib import Path
import einops
import torch
from torch import Tensor
import math
import numpy as np
import torch.nn.functional as F
from torch import nn
import torchtext
from torchtext import data as torchdata
from dataclasses import dataclass
from typing import Any, List
import random
import tensorboardX

N_HEADS = 8
D_HEAD = 64
MAX_SEQ = 100
D_MODEL = 512
N_WARMUP = 1000
N_LAYERS = 6
GPU = 0
BATCH_SIZE = 50
RUN_NAME = "run2"
LR_REDUCTION = 10.0

# N_LAYERS = 1
# D_MODEL = 6
# N_HEADS=1

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda', GPU)

def ar_mask(x: Tensor):
    d = x.size(2)
    batch_size = x.size(0)
    n_heads = x.size(1)
    mask = torch.ones(d, d).byte()
    mask = mask.tril()
    mask = mask.expand(batch_size, n_heads, -1, -1)
    mask = mask.to(device)
    out = torch.where(mask, x, torch.tensor(float("-inf")).to(device))
    return out


def compute_attention(Q, K, V, ar=False):
    batch_size, n_heads, n_words, dim = Q.shape
    s = torch.einsum("qhbd, qhcd->qhbc", Q, K)  # s[batch, head, query index, key index]
    s = s / math.sqrt(dim)
    if ar:
        s = ar_mask(s)
    s = F.softmax(s, dim=3)
    s = torch.einsum("qhbc,qhcd->qhbd", s, V)
    
        
    return s


def norm_layer():
    return lambda x: x  # TODO replace this with layer norm


class MultiHeadAttention(nn.Module):
    @staticmethod
    def create_head(*size):
        return nn.Parameter(1e-3 * torch.randn(*size))

    def __init__(self, n_heads, d, d_head):
        super().__init__()
        self.Wq = self.create_head(n_heads, d, d_head)
        self.Wk = self.create_head(n_heads, d, d_head)
        self.Wv = self.create_head(n_heads, d, d_head)
        self.Wo = self.create_head(n_heads * d_head, d)

    def forward(self, Q, K, V, ar=False):
        Qhead = torch.einsum("btd, hde->bhte", Q, self.Wq)
        Khead = torch.einsum("btd, hde->bhte", K, self.Wk)
        Vhead = torch.einsum("btd, hde->bhte", V, self.Wv)
        s = compute_attention(Qhead, Khead, Vhead, ar=ar)        
        s = einops.rearrange(s, "batch head time dim->batch (head dim) time")
        s = torch.einsum("bht,hd->btd", s, self.Wo)
        return s



class FeedForward(nn.Module):
    def __init__(self, d, d_inner=None):
        super().__init__()
        if d_inner is None:
            d_inner = 4 * d
        self.w1 = nn.Linear(d, d_inner)
        self.w2 = nn.Linear(d_inner, d)

    def forward(self, x):
        x = self.w1(x)
        x = F.relu(x)
        x = self.w2(x)
        return x

def dropout_layer():
    return nn.Dropout(p=.1)
    # return lambda x: x

class EncoderLayer(nn.Module):
    def __init__(self, d, n_heads, d_head):
        super().__init__()
        self.w = FeedForward(d)
        self.norm1 = norm_layer()
        self.norm2 = norm_layer()
        self.attention = MultiHeadAttention(n_heads=n_heads, d=d, d_head=d_head)
        self.dropout = dropout_layer()

    def forward(self, x):
        x = self.norm1(self.dropout(self.attention(x, x, x))) + x
        x = self.norm2(self.dropout(self.w(x))) + x
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d, n_heads, d_head):
        super().__init__()
        self.attention = MultiHeadAttention(n_heads=n_heads, d=d, d_head=d_head)
        self.w = FeedForward(d)
        self.norm1 = norm_layer()
        self.norm2 = norm_layer()
        self.norm3 = norm_layer()
        self.dropout = dropout_layer()

    def forward(self, dec_state, enc_state):
        x = (
            self.norm1(self.dropout(self.attention(dec_state, dec_state, dec_state, ar=True))) 
            + dec_state
        )
        attention = self.attention(x, enc_state, enc_state)
        x = self.norm2(self.dropout(attention)) + x
        x = self.norm3(self.dropout(self.w(x))) + x
        return x


class Decoder(nn.Module):
    def __init__(self, n_vocab, n_hidden, n_layers):
        super().__init__()
        self.attention_block = nn.ModuleList([
            DecoderLayer(n_hidden, N_HEADS, D_HEAD) for _ in range(n_layers)
        ])
        self.w_output = nn.Linear(n_hidden, n_vocab)
        self.embedding = nn.Embedding(n_vocab, n_hidden)
        self.position_encoder = PositionEncoder(d=n_hidden, seq_len=MAX_SEQ)
        self.dropout = dropout_layer()

    def forward(self, x, enc_state):
        x = self.embedding(x)
        x = self.position_encoder(x)
        x = self.dropout(x)
        for i, layer in enumerate(self.attention_block):
            x = layer(x, enc_state)
        x = self.w_output(x)
        return x


class Encoder(nn.Module):
    def __init__(self, n_vocab, n_hidden, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(n_vocab, n_hidden)
        self.attention_block = nn.ModuleList([
            EncoderLayer(n_hidden, N_HEADS, D_HEAD) for _ in range(n_layers)
        ])
        self.position_encoder = PositionEncoder(d=n_hidden, seq_len=MAX_SEQ)
        self.dropout = dropout_layer()

    def forward(self, x):
        x = self.embedding(x)
        x = self.position_encoder(x)
        x = self.dropout(x)
        for layer in self.attention_block:
            x = layer(x)
        return x


class EncoderDecoder(nn.Module):
    def __init__(self, n_vocab, n_hidden, n_layers):
        super().__init__()
        self.encoder = Encoder(n_vocab, n_hidden, n_layers)
        self.decoder = Decoder(n_vocab, n_hidden, n_layers)

    def forward(self, input_tokens, output_tokens):
        enc_state = self.encoder(input_tokens)
        decoder_output = self.decoder(output_tokens, enc_state)
        return decoder_output


class PositionEncoder(nn.Module):
    def __init__(self, d: int, seq_len: int):
        super().__init__()
        pos = torch.arange(seq_len, dtype=torch.float32)
        d_range = torch.arange(0, d, step=2, dtype=torch.float32)
        wavelength = torch.exp(-d_range / d * math.log(10000.0))
        x = torch.einsum("t,d->td", pos, wavelength)
        out = torch.empty(seq_len, d)
        out[:, ::2] = torch.sin(x)
        out[:, 1::2] = torch.cos(x)
        self.register_buffer('encodings', out)
        self.encodings = out

    def forward(self, x):
        n_seq = x.size(1)
        encodings = self.encodings[:n_seq, :].unsqueeze(0)
        out = x + encodings
        return out


@dataclass
class Data:
    dataset: Any
    batcher: Any
    field_in: Any
    field_out: Any


def load_data(batch_size=BATCH_SIZE):
    field_in = torchdata.Field(tokenize="spacy", batch_first=True)
    field_out = torchdata.Field(
        tokenize="spacy", init_token="<sos>", eos_token="<eos>", batch_first=True
    )
    dataset = torchtext.datasets.TranslationDataset(
        "data/", ("examples.txt", "answers.txt"), (field_in, field_out)
    )
    field_in.build_vocab(dataset)
    field_out.build_vocab(dataset)
    bucket_iter = torchdata.BucketIterator(dataset, batch_size, sort_key=lambda x: x.trg)
    return Data(dataset, bucket_iter, field_in, field_out)

def schedule_lr(epoch: int, d: int, n_warmup: int = 100) -> float:
    epoch += 1
    ramp = epoch * n_warmup ** -1.5
    asym = epoch ** -.5
    lr = d ** -.5 * min(ramp, asym)
    lr /= LR_REDUCTION
    return lr

def get_model_dir():
    p = Path.home() / 'models'
    p.mkdir(exist_ok=True)
    return p

def save_model(m):
    torch.save(m.state_dict(), get_model_dir() / "transformer.model")
    
def get_vocab_size(data):
    return max(len(data.field_in.vocab), len(data.field_out.vocab))

def load_model(data=None):
    if data is None:
        data = load_data()
    m = create_model(data)
    state_dict = torch.load(get_model_dir() / 'transformer.model', map_location=device)
    m.load_state_dict(state_dict)
    return m

def create_model(data):
    n_vocab = get_vocab_size(data)
    model = EncoderDecoder(n_vocab, D_MODEL, N_LAYERS)
    return model

def train(data=None, n_epochs=50000, model=None):
    if data is None:
        data = load_data()
    if model is None:
        model = create_model(data)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters())
    writer = tensorboardX.SummaryWriter(log_dir=str(Path.home()/'runs'/'transformer'/RUN_NAME))
    try:
        step = 0
        running_loss = torch.tensor(10.0)
        for epoch in range(n_epochs):
            print(f"Running loss is {running_loss} on epoch {epoch}")

            save_model(model)
            if epoch % 100 == 0:
                torch.save(model.state_dict(), get_model_dir()/f"transformer_{epoch}.model")
            for batch in data.batcher:
                lr = schedule_lr(step, d=D_MODEL, n_warmup=N_WARMUP)
                for group in optim.param_groups:
                    group['lr'] = lr
                x, y = batch.src.to(device), batch.trg.to(device)
                y_in = y[:, :-1]
                y_out = y[:, 1:]
                y_pred = model(x, y_in)
                y_pred = y_pred.permute(0, 2, 1)
                loss = loss_fn(y_pred, y_out)
                running_loss = .99 * running_loss + .01*loss
                writer.add_scalar("loss", loss, global_step=step)
                step += 1
                optim.zero_grad()
                loss.backward()
                optim.step()
            # if running_loss < 1e-3:
            #     break
    except KeyboardInterrupt:
        print("Breaking because of interrupt")
    return model


def generate(question: str, data: Data, model: EncoderDecoder, stop_on_eos=False) -> str:
    model.train(False)    
    f = data.field_in
    tokens = f.tokenize(question)
    x = f.numericalize([tokens]).to(device)
    y = torch.tensor(data.field_out.vocab.stoi[data.field_out.init_token])[None, None].to(device)
    words:List[int] = []
    eos_reached = False
    eos_token = data.field_out.vocab.stoi[data.field_out.eos_token]
    for i in range(25):
        if eos_reached and stop_on_eos:
            break
        with torch.no_grad():
            pred = model(x, y)
        pred = pred[0, -1, :]
        
        d = torch.distributions.Categorical(logits=F.log_softmax(pred, dim=0))
        word = d.sample()
        words.append(word.item())
        if word == eos_token:
            eos_reached = True
            if stop_on_eos:
                words.pop()  # Remove the EOS token
                break
        word = word[None, None]
        y = torch.cat([y, word], dim=1)
    str_words = [data.field_out.vocab.itos[word] for word in words]
    return ' '.join(str_words)

def generate_samples(n_samples:int):
    words = "jon sarah so cute love forever perfect harmony good great is".split()
    in_file = open('data/examples.txt', 'w')
    out_file = open('data/answers.txt', 'w')
    for i in range(n_samples):
        n_words = int(torch.randint(3, 15, size=()))
        sentence = []
        for pos in range(n_words):
            word_idx = int(torch.randint(0, len(words), size=()))
            sentence.append(words[word_idx])
        joined_sentence_p1 = ' '.join(sentence[:-2])
        joined_sentence_p2 = ' '.join(sentence[-2:])
        response = f"Good to hear that {joined_sentence_p1}. But also {joined_sentence_p2}!"
        in_file.write(' '.join(sentence))
        in_file.write("\n")
        out_file.write(response)
        # out_file.write(joined_sentence)
        out_file.write("\n")
    return

def generate_qa(n_samples: int):
    in_file = open('data/examples.txt', 'w')
    out_file = open('data/answers.txt', 'w')
    male_names = ["Jon", "Todd", "Simon", "Owen"]
    female_names = ["Sarah", "Lily", "Tamara", "Marianne"]
    all_names = male_names + female_names
    years = []
    for i in range(100):
        years.append(str(1700+10*i))
    a_objs = ["ball", "car", "house"]
    an_objs = ["antelope", "elephant", "emotion"]
    person_type = ["pop star", "scientist", "artist", "friend", "lover"]
    thing_types = ["thing", "person", "idea", "lover", "cuddle bunny", "loving thing"]
    for i in range(n_samples):
        q_type = random.choice(["what", "who", "when"])
        if q_type == "what":
            noun_type = random.choice(["a", "an"])
            if noun_type == "a":
                obj = random.choice(a_objs)
            else:
                obj = random.choice(an_objs)
            question = f"What is {noun_type} {obj}?"
            thing_type = random.choice(thing_types)
            answer = f"{noun_type.capitalize()} {obj} is a kind of {thing_type}."
        elif q_type == "who":
            gender = random.choice(["male", "female"])
            if gender == "male":
                name = random.choice(male_names)
                prefix = "mr"
            else:
                name = random.choice(female_names)
                prefix = "ms"
            question = f"Who is {name}?"
            ptype = random.choice(person_type)
            answer = f"{prefix.capitalize()}. {name} is a {ptype}."
        elif q_type == "when":
            name = random.choice(all_names)
            question = f"When was {name} born?"
            year = random.choice(years)
            answer = f"{name} was born in {year}!"
        else:
            raise Exception(f"Unknown q_type {q_type}")
        in_file.write(f"{question}\n")
        out_file.write(f"{answer}\n")
