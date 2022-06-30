

# import pandas as pd
# df = pd.read_excel('Documents/Datalar.xlsx') 
# messages = df['Mesaj İçeriği'].tolist()

# messages = [m for m in messages if str(m) != 'nan']

# filteredSent = [fs(s) for s in messages]

# model = w2v(filteredSent, 0)
# model = w2v(filteredSent, 5)
# model = w2v(filteredSent, 10)
# model = w2v(filteredSent, 20)

from gensim.models import Word2Vec
model1=Word2Vec.load('w2v/w2v_word10_size300_(784).model')

s1="ziraat mühendisi öğrencisiyim staj yapmak istiyorum"
s2="hastalık mıdır "
distance = model1.wv.n_similarity(s1.lower().split(), s2.lower().split())
print(distance)

model=Word2Vec.load('w2v/w2v_word10_size300_(784).model')
s1 = 'bitki koruma ürünlerini göster'
s2 = 'hangi ilaç bitkime iyi gelir'
wmdistance=model.wv.wmdistance(s1.lower().split(), s2.lower().split())
print("Wmdistance:", wmdistance)



import math
import re
from collections import Counter

WORD = re.compile(r"\w+")

def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)

s1 = 'the dog bites the man '
s2 = 'the man bites the dog'

vector1 = text_to_vector(s1)
vector2 = text_to_vector(s2)

cosine = get_cosine(vector1, vector2)

print("Cosine:", cosine)
