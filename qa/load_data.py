from dataclasses import dataclass
from typing import List, Sequence, Any
import json
import read_articles
import random

"""
Most of the code was taken from here :  https://github.com/rodgzilla/pytorch-pretrained-BERT/tree/multiple-choice-code/examples

John's edited the code first and then then Stav and Nurit changed it to run on RoBerta and not BERT

"""

@dataclass()
class OnestopId:
    article_id: int
    paragraph_id: int
    level: str

@dataclass
class Document:
    answers: List[int]
    options: List[List[str]]
    questions: List[str]
    article: str
    id: Any

@dataclass()
class QuestionId:
    paragraph_id: OnestopId
    question: int

class Example(object):
    """A single training/test example for the SWAG dataset."""

    def __init__(self,
                 id,
                 context_sentence,
                 start_ending,
                 ending_0,
                 ending_1,
                 ending_2,
                 ending_3,
                 label=None):
        self.id = id
        self.context_sentence = context_sentence
        self.start_ending = start_ending
        self.endings = [
            ending_0,
            ending_1,
            ending_2,
            ending_3,
        ]
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            "id: {}".format(self.id),
            "context_sentence: {}".format(self.context_sentence),
            "start_ending: {}".format(self.start_ending),
            "ending_0: {}".format(self.endings[0]),
            "ending_1: {}".format(self.endings[1]),
            "ending_2: {}".format(self.endings[2]),
            "ending_3: {}".format(self.endings[3]),
        ]

        if self.label is not None:
            l.append("label: {}".format(self.label))

        return ", ".join(l)



def parse_race(train_dir,is_training):
    docs = []
    answer_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    for path in train_dir.iterdir():
        parsed = json.loads(path.read_bytes())
        answers = [answer_map[_] for _ in parsed['answers']]
        example = Document(answers=answers, options=parsed['options'], questions=parsed['questions'],
                           article=parsed['article'], id=parsed['id'])
        docs.append(example)
    examples = split_documents(docs,is_training)
    return examples



def split_documentsOriginal(docs: Sequence[Document]): # original form johns code, the problem is that all the labels are 0 and might bias the model
    examples = []
    for doc in docs:
        n_questions = len(doc.questions)
        for i in range(n_questions):
            example = Example(id=QuestionId(paragraph_id=doc.id, question=i), context_sentence=doc.article, start_ending=doc.questions[i],
                              ending_0=doc.options[i][0], ending_1=doc.options[i][1], ending_2=doc.options[i][2],
                              ending_3=doc.options[i][3], label=doc.answers[i])
            #print(doc.answers[i])
            examples.append(example)
    return examples

def split_documents(docs: Sequence[Document], is_training):  # randomizing the questions order to undo the bias
        examples = []
        j = 0
        if is_training:
            for doc in docs:
                n_questions = len(doc.questions)
                for i in range(n_questions):
                    if j == 0:
                        example = Example(id=QuestionId(paragraph_id=doc.id, question=i), context_sentence=doc.article,
                                          start_ending=doc.questions[i],
                                          ending_0=doc.options[i][0], ending_1=doc.options[i][1],
                                          ending_2=doc.options[i][2],
                                          ending_3=doc.options[i][3], label=0)
                        j += 1
                    elif j == 1:
                        example = Example(id=QuestionId(paragraph_id=doc.id, question=i), context_sentence=doc.article,
                                          start_ending=doc.questions[i],
                                          ending_0=doc.options[i][1], ending_1=doc.options[i][0],
                                          ending_2=doc.options[i][2],
                                          ending_3=doc.options[i][3], label=1)
                        j += 1
                    elif j == 2:
                        example = Example(id=QuestionId(paragraph_id=doc.id, question=i), context_sentence=doc.article,
                                          start_ending=doc.questions[i],
                                          ending_0=doc.options[i][2], ending_1=doc.options[i][1],
                                          ending_2=doc.options[i][0],
                                          ending_3=doc.options[i][3], label=2)
                        j += 1
                    elif j == 3:
                        example = Example(id=QuestionId(paragraph_id=doc.id, question=i), context_sentence=doc.article,
                                          start_ending=doc.questions[i],
                                          ending_0=doc.options[i][3], ending_1=doc.options[i][1],
                                          ending_2=doc.options[i][2],
                                          ending_3=doc.options[i][0], label=3)
                        j = 0
                    examples.append(example)
        else:
            for doc in docs:
                n_questions = len(doc.questions)
                for i in range(n_questions):
                    example = Example(id=QuestionId(paragraph_id=doc.id, question=i), context_sentence=doc.article,
                                      start_ending=doc.questions[i],
                                      ending_0=doc.options[i][0], ending_1=doc.options[i][1],
                                      ending_2=doc.options[i][2],
                                      ending_3=doc.options[i][3], label=doc.answers[i])

                    examples.append(example)
        return examples



def read_race_examples(input_file, is_training):
    return parse_race(input_file,is_training)

def read_onestop_docs():
    articles = read_articles.read_all_articles(read_articles.ANNOTATIONS_FOLDER)
    docs = []
    for article in articles:
        for annotation in article.paragraph_annotations:
            for paragraph in annotation.paragraph_versions:
                questions = []
                options = []
                for question in annotation.questions:
                    answers = []
                    questions.append(question.question)
                    for answer in question.answers:
                        answers.append(answer)
                    options.append(answers)
                onestop_id = OnestopId(article_id=article.article_id, paragraph_id=annotation.paragraph_id, level=paragraph.level)
                doc = Document(answers=[0]*4, options=options, questions=questions, article=paragraph.plain_text, id=onestop_id)
                docs.append(doc)
    return docs

def get_article_id(example: Example):
    return example.id.paragraph_id.article_id

def read_onestop(is_training=True, return_all=False)->List[Example]: # 4Folded edition
    docs = read_onestop_docs()
    examples =  split_documents(docs,is_training)
    if return_all:
        return examples
    article_ids = list(set([get_article_id(_) for _ in examples]))
    last_state = random.getstate()
    random.seed(0, version=2)
    random.shuffle(article_ids)

    n_07 = round(.7*len(article_ids))
    n_03= round(0.3 * len(article_ids))
    n_01 = round(0.1 * len(article_ids))
    n_08=round(0.8 * len(article_ids))
    n_06=round(0.6 * len(article_ids))



    #### fold 1 original
    n_train = round(.7*len(article_ids))
    articles_train = set(article_ids[:n_train])
    articles_test = set(article_ids[n_train:])
    

    """
    #####fold 2
    articles_train = set(article_ids[n_03:])
    articles_test = set(article_ids[:n_03])
    #### end fold 2
   
    
    ####### fold 3
    
    articles_train = set(article_ids[:n_03])  # first 10%
    articles_train.update(article_ids[n_07:])  # 40% to 100%

    articles_test = set(article_ids[n_03:n_06])  # 10% to 40%

    """
    ####### fold 4
    """
    articles_train = set(article_ids[:n_06])  # first 10% to 80%
    articles_train.update(article_ids[n_07:n_08])  # 40% to 100%


    articles_test = set(article_ids[n_06:n_07])  # 0% to 10%
    """

    if is_training:
        res = [_ for _ in examples if get_article_id(_) in articles_train]
    else:
        res = [_ for _ in examples if get_article_id(_) in articles_test]
    random.setstate(last_state)
    return res




def read_onestopOrginal(is_training=True, return_all=False)->List[Example]: # original , 1 fold
    docs = read_onestop_docs()
    examples =  split_documents(docs,is_training)
    if return_all:
        return examples
    article_ids = list(set([get_article_id(_) for _ in examples]))
    last_state = random.getstate()
    random.seed(0, version=2)
    random.shuffle(article_ids)
    n_train = round(.7*len(article_ids))
    articles_train = set(article_ids[:n_train])
    articles_test = set(article_ids[n_train:])
    if is_training:
        res = [_ for _ in examples if get_article_id(_) in articles_train]
    else:
        res = [_ for _ in examples if get_article_id(_) in articles_test]
    random.setstate(last_state)
    return res

