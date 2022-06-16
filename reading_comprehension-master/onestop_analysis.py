import math

import pandas
import load_data
from pathlib import Path
import torch
import numpy as np
import plotly
import plotly.figure_factory as ff
import plotly.graph_objs as go
from plotly.offline import plot
import pickle

def load_results(use_cache=True):
    if use_cache:
        return pandas.read_pickle(Path(__file__).parent/'data/onestop_analysis.pickle')
    pickle_file = Path.home()/'models/race/eval_results.pickle'
    results = torch.load(pickle_file)
    results = np.concatenate(results, axis=0)
    questions = load_data.read_onestop()
    df = []
    p_norm = np.exp(results)/np.exp(results).sum(axis=1)[:, None]
    for i in range(len(questions)):
        q = questions[i]
        chosen_answer = np.argmax(results[i, :])
        df.append((q.id.paragraph_id.article_id, q.id.paragraph_id.paragraph_id, q.id.paragraph_id.level, q.id.question, results[i, :], results[i,0], chosen_answer==0, p_norm[i, 0], chosen_answer))
    df = pandas.DataFrame(df, columns=['article', 'paragraph', 'level', 'question', 'logits', 'logit_gold', 'correct', 'p_gold', 'answer'])
    df['level']= df['level'].astype('category')
    df['answer'] = pandas.Categorical.from_codes(df['answer'], ['A', 'B', 'C', 'D'])
    df = df.set_index(['article', 'paragraph', 'level', 'question'])
    return df

def accuracy_by_group(show_plot=True, save_plot=True):
    df = load_results()
    g = df.groupby(level='level').mean()
    f = go.Figure()
    levels = ['Ele', 'Int', 'Adv']
    bar_accuracy = go.Bar(x=levels, y=g.loc[levels, 'correct']*100, name='Accuracy')
    f = go.Figure()
    f.layout.yaxis.ticksuffix='%'
    f.layout.yaxis.title = 'Accuracy'
    f.layout.xaxis.title = 'Paragraph difficulty'
    f.layout.title = 'BERT accuracy vs paragraph difficulty on onestop'
    f.layout.title.font.size = 20
    f.layout.yaxis.range = [45, 55]
    f.add_trace(bar_accuracy)
    if show_plot:
        plot(f)
    if save_plot:
        plotly.io.write_image(f, 'figs/accuracy_by_group.pdf')
    return f

def confusion_matrix():
    df = load_results()
    counts = df.answer.value_counts(normalize=True)
    f = go.Figure()
    answers = ['A', 'B', 'C', 'D']
    f.add_bar(x=answers, y=counts[answers], name='Overall')
    level = df.index.get_level_values("level")
    xtab = pandas.crosstab(level, df.answer, normalize='index')
    for level in xtab.index:
        f.add_bar(x=answers, y=xtab.loc[level][answers], name=level)
    f.layout.xaxis.title = 'Chosen answer'
    f.layout.yaxis.title = 'Fraction of times chosen'
    f.layout.title = 'Chosen answer by difficulty level'
    plotly.io.write_image(f, 'figs/confusion.pdf')
    plot(f)

    return xtab

if __name__=="__main__":
    load_results()