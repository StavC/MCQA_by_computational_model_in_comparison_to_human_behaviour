## 100% John's code, didn't touch it.

from bs4 import BeautifulSoup
from qa.global_vars import *
def sublist_inds(l, sub_l):
    span_len = len(sub_l)
    for i in range(len(l)):
        if l[i:i+span_len] == sub_l:
            return (i, i+span_len)
    raise ValueError("No span found", str(l), str(sub_l))

class Question():
    def __init__(self, annotated_string, repeats_span = None):
        self.question = ""
        self.answers = []
        self.repeats_span = repeats_span
        self.parse_question(annotated_string)
        
    def parse_question(self, annotated_string):
        parts = annotated_string.strip().split("\n")
        assert(len(parts) == 5)
        assert(parts[0].startswith("Q: ") or parts[0].startswith("Q1: ") or parts[0].startswith("Q2: "))
        if parts[0].startswith("Q: "):
            self.question = parts[0][3:]
        else:
            self.question = parts[0][4:]
        for ind, ans in zip(range(1,5), "abcd"):
            assert parts[ind].startswith(ans+": "), parts[ind]
            self.answers.append(parts[ind][3:])

class Paragraph():
    def __init__(self, annotated_string, num_questions, level = ""):
        self.num_questions = num_questions
        self.level = level
        self.annotated_string = annotated_string.strip().replace("\n", ' ')
        self.plain_text = ""
        self.A_spans, self.A_inds = [], []
        self.D_spans, self.D_inds = [], []
        self.parse_paragraph(self.annotated_string)

    def parse_paragraph(self, annotated_string):
        soup =  BeautifulSoup(annotated_string, 'lxml')
        #add plain text
        self.plain_text = soup.get_text()
        #add answer spans
        for q_id in range(1, 1 + self.num_questions):
            A = [span.get_text() for span in soup.find_all("a"+str(q_id))]
            self.A_spans.append(A)
            A_inds = [sublist_inds(self.plain_text.split(), s.split()) for s in A]
            self.A_inds.append(A_inds)
            D = [span.get_text() for span in soup.find_all("d"+str(q_id))]
            self.D_spans.append(D)
            D_inds = [sublist_inds(self.plain_text.split(), s.split()) for s in D]
            self.D_inds.append(D_inds)
    
class ParagraphAnnotation():
    def __init__(self, annotated_paragraph, paragraph_id = None, article_id = None, article_title = None):
        self.paragraph_id = paragraph_id
        self.article_id = article_id
        self.article_title = article_title
        self.paragraph_versions = [] #3 Paragraphs, one for each version
        self.questions = [] #Questions
        self.parse_annotations(annotated_paragraph)

    def parse_annotations(self, annotated_paragraph):
        parts = annotated_paragraph.strip().split("\n\n")
        #print("=====")
        #for p in parts:
        #    print("---", p[:30], "---")
        if (len(parts) != 6):
            for part in parts:
                print(part)
                print ("--------")
        assert(len(parts) == 6)  #3 paragraph version + 3 questions
        num_questions = 3
        
        #read paragraph versions
        for ind, level in zip(range(3), ["Adv", "Int", "Ele"]):
            assert parts[ind].startswith(level + ": "), parts[ind]
            version_str = parts[ind][5:]
            paragraph = Paragraph(version_str, num_questions, level = level)
            self.paragraph_versions.append(paragraph)

        #read first two questions (distinct A spans)
        for q_anno_str in parts[3:5]:
            assert q_anno_str.startswith("Q: "), q_anno_str
            q = Question(q_anno_str)
            self.questions.append(q)
        assert len(self.questions) == 2, len(self.questions)

        #read third question (same A span as one of the first two)
        q_anno_str = parts[5]
        assert q_anno_str.startswith(("Q1: ", "Q2: ")), q_anno_str
        q = Question(q_anno_str, repeats_span = 0)
        if q_anno_str.startswith("Q1: "):
            q.repeats_span = 0
            self.questions[0].repeats_span = 2
            self.questions.append(q)
        else:
            q.repeats_span = 1
            self.questions[1].repeats_span = 2
            self.questions.append(q)
        assert len(self.questions) == 3, len(self.questions)

class Article():
    def __init__(self, annotation_path, article_id = None):
        self.title = ""
        self.article_id = article_id
        self.paragraph_annotations = []
        self.num_paragraphs = 0
        self.parse_annotation_file(annotation_path)
        
    def parse_annotation_file(self, annotation_path):
        f = open(annotation_path, 'r', encoding='utf-8')
        f_str = f.read()
        f.close()
        title_anno, rest = f_str.split("\n\n\n")
        self.title = title_anno.split("\n")[1].strip()
        paragraph_strs = rest.split("# Paragraph\n\n")[1:]
        #assert len(paragraph_strs) > 0
    
        for ind, p in enumerate(paragraph_strs):
            self.paragraph_annotations.append(ParagraphAnnotation(p, paragraph_id = ind+1, article_id = self.article_id, article_title = self.title))
        self.num_paragraphs = len(self.paragraph_annotations)

def read_all_articles(articles_dir):
    all_articles = []
    for ind, article_path in enumerate(articles_dir.iterdir()):
        a = Article(article_path, article_id = ind+1)
        all_articles.append(a)
    return all_articles

if __name__ == "__main__":
    read_all_articles(ANNOTATIONS_FOLDER)
