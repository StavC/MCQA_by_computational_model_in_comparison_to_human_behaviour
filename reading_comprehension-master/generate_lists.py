import csv
import string
import random
import numpy as np
import pandas as pd
from global_vars import *
from read_articles import read_all_articles


#index tuples of (difficulty level, question) 
conditions_map = {0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2), 6:(2,0), 7:(2,1), 8:(2,2)}

def is_latin_grid(square, size, repeat_rows = 1, repeat_cols = 1):
    n_rows, n_cols = square.shape
    for i in range(size):
        assert all(((square == np.ones((n_rows, n_cols))*i).sum(axis=1)) == repeat_cols)
        assert all(((square == np.ones((n_rows, n_cols))*i).sum(axis=0)) == repeat_rows)
    return True

def create_latin_square(n):
    symbols = range(n)
    square = np.array([[symbols[i - j] for i in range(n)] for j in range(n, 0, -1)])
    #shuffle rows
    np.random.shuffle(square)
    #shuffle cols
    square = square.transpose()
    np.random.shuffle(square)
    assert(is_latin_grid(square, n))
    return square

def create_latin_grid(n, n_trials, n_lists):
    h_blocks = []
    for list_block in range(n_lists//n):
        v_blocks = []
        for trial_block in range(n_trials//n):
            v_blocks.append(create_latin_square(n))
        v_grid = np.vstack(v_blocks)
        assert is_latin_grid(v_grid, n, repeat_rows = n_trials//n)
        h_blocks.append(v_grid)
    grid = np.hstack(h_blocks)
    assert is_latin_grid(grid, n, repeat_rows = n_trials//n, repeat_cols = n_lists//n)
    print(grid)
    return grid

def generate_dat_file(parags, grid, answers_order):
    n_trials = grid.shape[0]
    parag_ids = [p.paragraph_id for p in parags]
    article_ids = [p.article_id for p in parags]
    article_titles = [p.article_title for p in parags]
    all_dfs = []
    for col_ind in range(grid.shape[1])[:1]:
        list_dfs = []
        for preview in [1]:
            df = pd.DataFrame()
            df["list"] = [col_ind+1 for i in range(n_trials)]
            df["has_preview"] = preview
            df["trial"] = [i+1 for i in range(n_trials)]
            df["paragraph_id"] = parag_ids
            df["article_id"] = article_ids
            df["article_title"] = article_titles
            df["paragraph_level"] = [parags[ind].paragraph_versions[conditions_map[item][0]].level for ind, item in enumerate(grid[:,col_ind])]
            df["paragraph"] = [parags[ind].paragraph_versions[conditions_map[item][0]].plain_text for ind, item in enumerate(grid[:,col_ind])]

            #df["qestion"] = [parags[ind].questions[conditions_map[item][1]] for ind, item in enumerate(grid[:,col_ind])]
            df["question_id"] = [conditions_map[item][1] for ind, item in enumerate(grid[:,col_ind])]
            df["question"] = [parags[ind].questions[conditions_map[item][1]].question for ind, item in enumerate(grid[:,col_ind])]
            for choice_ind, choice in enumerate(["a", "b", "c", "d"]):
                df[choice] = [parags[ind].questions[conditions_map[item][1]].answers[answers_order[conditions_map[item][1]][ind][choice_ind]] for ind, item in enumerate(grid[:,col_ind])]
            df["correct_answer"] = [list(answers_order[conditions_map[item][1]][ind]).index(0) \
                                    for ind, item in enumerate(grid[:,col_ind])]
            df["answers_order"] = [answers_order[conditions_map[item][1]][ind] \
                                   for ind, item in enumerate(grid[:,col_ind])]
            df["Aspan_inds"] = [parags[ind].paragraph_versions[conditions_map[item][0]].A_inds[conditions_map[item][1]] for ind, item in enumerate(grid[:,col_ind])]
            df["Dspan_inds"] = [parags[ind].paragraph_versions[conditions_map[item][0]].D_inds[conditions_map[item][1]] for ind, item in enumerate(grid[:,col_ind])]
           
            list_dfs.append(df)
            merged_df = pd.concat(list_dfs)
        all_dfs.append(merged_df)
    final_df = pd.concat(all_dfs) 
    print(final_df)
    return final_df

def generate_answers_order(n_question_types, n_trials, n_answers):
    all_answers = []
    answers = range(n_answers)
    for i in range(n_question_types):
        answer_order = np.vstack([random.sample(answers, n_answers) for q in range(n_trials)])
        all_answers.append(answer_order)
    return all_answers

if __name__ == "__main__":
    n_conditions = 9
    n_lists = 18
    articles = read_all_articles(ANNOTATIONS_FOLDER)
    
    total_paragraphs = sum((a.num_paragraphs for a in articles))
    #remove this

    n_trials = (total_paragraphs//n_conditions)*n_conditions


    assert((n_trials % n_conditions == 0) and (n_lists % n_conditions == 0))

    print("trials", n_trials, "lists", n_lists, "conditions", n_conditions)

    parags = [p for a in articles for p in a.paragraph_annotations][:n_trials]

    #read this from file?
    grid = create_latin_grid(n_conditions, n_trials, n_lists)
    n_answers = 4
    n_question_types = 3
    #read this from file?
    answers_order = generate_answers_order(n_question_types, n_trials, n_answers)

    df = generate_dat_file(parags, grid, answers_order)
    
    df.to_csv(DAT_FOLDER+'onestop.dat', sep='\t', index = False, quoting=csv.QUOTE_NONNUMERIC, encoding = "utf-8")

