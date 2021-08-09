from gensim.models import FastText
from gensim.test.utils import common_texts  # some example sentences

print(common_texts[0])
['human', 'interface', 'computer']
print(len(common_texts))
9
model = FastText(vector_size=4, window=3, min_count=1)  # instantiate
model.build_vocab(sentences=common_texts)
model.train(sentences=common_texts, total_examples=len(common_texts), epochs=10) 

model["Pouvez-vous recommander un restaurant chinois à Manhattan?"]