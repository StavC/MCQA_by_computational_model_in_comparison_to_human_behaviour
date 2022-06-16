import pytest
import transformer as a
import torch
from torch import as_tensor as tensor
import plotly
from plotly.offline import plot
from plotly import graph_objs as go

def test_prepare():
    y = tensor([[1, 2], [3, 4]])
    a.prepare_batch(y)

def test_train():
    a.train(n_epochs=2)


def test_position():
    e = a.PositionEncoder(d=50, seq_len=10)
    f = go.Figure()
    f.add_heatmap(z=e.encodings)
    f.layout.xaxis.title = 'Dimension'
    f.layout.yaxis.title = 'Position'
    plot(f)
    x = torch.randn(2, 5, 50)
    y = e(x)
    return


def test_gen():
    a.generate_samples()


def test_jdebug():
    data = a.load_data()
    a.jdebug(data)
