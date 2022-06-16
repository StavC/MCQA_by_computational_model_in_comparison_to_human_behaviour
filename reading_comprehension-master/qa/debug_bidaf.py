from . import bidaf

examples = bidaf.read_training()
a = bidaf.tokenize_all(examples, 50)
b=bidaf.BidafBert(10)
y=b(a.context, a.context_mask, a.context_seq, a.query, a.query_mask, a.query_seq)

