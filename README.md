This Project and Paper were authored by Stav Cohen and Nurit Klimovitsky Maor.

Contact:

cohnstav@campus.technion.ac.il
knurit@campus.technion.ac.il
Acknowledgments:
Some of the code utilized in this project was based on a pre-existing legacy version provided and developed by Dr. Yevgeni Berzak.

Abstract
Multiple Choice Question Answering (MCQA) is a commonly employed method to assess reading comprehension in both humans and language models. The task of MCQA, based on a given contextual text, presents a significant challenge for language models.

In traditional Natural Language Processing (NLP) research, the primary objective is to develop models that achieve high accuracy in selecting the correct answer.

In this study, our goal is to compare the question-answering capabilities of a computational model with observed human behavior. We fine-tuned a RoBERTa model using the RACE (Lai et al., 2017) and OneStopQA (Berzak et al., 2020) datasets. Subsequently, we applied the fine-tuned model to the OneStopQA dataset and obtained prediction distributions for each question.

The observed human behavior data utilized in this study was collected in (Berzak et al., 2020) through the crowd-sourcing platform Prolific (Pro). We present the results and analysis of the comparison between the model's predictions and human responses.

Motivation
A computational model trained to perform the MCQA task with a level of similarity to observed human behavior can have significant implications in various domains:

Reducing reliance on data gathered through human surveys and studies by using the model's predictions as substitutes for human responses.
Identifying flawed questions used in assessment tests (e.g., SAT or other reading comprehension exams).
Assessing the difficulty of text and questions.
Evaluating text simplification by comparing the model's predictions across different levels of contextual complexity.




