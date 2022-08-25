# MCQA_by_computational_model_in_comparison_to_human_behaviour

This Project and Paper were created by Stav Cohen And Nurit Klimovitsky Maor



cohnstav@campus.technion.ac.il
knurit@campus.technion.ac.il

we based some of our code on a pre existing legacy version that were given and created by Dr. Yevgeni Berzak.



# Abstract 

Multiple Choice Question Answering (MCQA)
is widely used to study and measure reading
comprehension in humans and in language
models. The task of MCQA based on a given
context text is a challenging task for language
models. In traditional NLP research the main
goal is to achieve models that select the correct
answer with high accuracy scores. In this work
we aim to compare the question answering abilities of a computational model to observed human behaviour. We utilize a RoBERTa model
by fine-tuning it on RACE (Lai et al., 2017) and
OneStopQA (Berzak et al., 2020). We run the
fine-tuned model on OneStopQA and obtain
the prediction distribution for each question.
The data that is used as the observed human behaviour is the data that was gathered in (Berzak
et al., 2020) using the crowd-sourcing platform
Prolific (Pro). In this work we show the results
analysis of the comparison between the model
and the human responses


# Motivation

A computational model that is trained
to perform the MCQA task with sufficient similarity to the observed human behaviour can have
paramount influence on many fields.
• Reducing the dependencies in data gathered
by human surveys and studies. It can be done
by using the model to get predictions that replace the need for human response.
• Identifying faulty questions that are used for
assessment tests (such as SAT test or any other
reading comprehension exams).
• Difficulty assessment of text and questions.
• Assessment of text simplification by comparing the predictions of a model in different
levels of the context text.
