import re
import gensim 
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from gensim.models import Word2Vec 

df = pd.read_excel('Documents/Datalar.xlsx') 
messages = df['Mesaj İçeriği'].tolist()

messages = [m for m in messages if str(m) != 'nan']

filtered_messages = [re.sub(r"['].*?[^\w]", " " , row) for row in messages]
filtered_messages =[re.sub("[^a-zA-ZığüşöçİĞÜŞÖÇ ']"," ", row)for row in messages]

filtered_messages = [str(row).lower().split() for row in filtered_messages]

from nltk.corpus import stopwords
stop_words = stopwords.words('turkish')

cleaned_messages=[]
for row in filtered_messages:
    clean = [word for word in row if word not in stop_words]
    cleaned_messages.append(clean)
    
model = gensim.models.Word2Vec(cleaned_messages, min_count = 15, 
 							size = 300, window = 5) 
vocablist=model.wv.vocab

with open('Documents/word2vec.txt', 'w', encoding='utf-8') as f:
    for item in vocablist:
        f.write("%s\n" % item)

words_wp = []
embeddings_wp = []    
for word in list(model.wv.vocab):
    embeddings_wp.append(model.wv[word])
    words_wp.append(word)
    
tsne_wp_3d = TSNE(perplexity=30, n_components=3, init='pca', n_iter=3500, random_state=12)
embeddings_wp_3d = tsne_wp_3d.fit_transform(embeddings_wp)

from mpl_toolkits.mplot3d import Axes3D

def tsne_plot_3d(title, label, embeddings, a=1):
    fig = plt.figure()
    ax = Axes3D(fig)
    colors = cm.rainbow(np.linspace(0, 1, 1))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], c=colors, alpha=a, label=label)
    plt.legend(loc=4)
    plt.title(title)
    plt.show()

tsne_plot_3d('Visualizing Embeddings using t-SNE', ' ', embeddings_wp_3d, a=0.1)

# def display_closestwords_tsnescatterplot(model, vocablist):
    
#     arr = np.empty((0,300), dtype='f')
#     word_labels = [vocablist]

#     # get close words
#     close_words = model.similar_by_word(vocablist)
    
#     # add the vector for each of the closest words to the array
#     arr = np.append(arr, np.array([model[vocablist]]), axis=0)
#     for wrd_score in close_words:
#         wrd_vector = model[wrd_score[0]]
#         word_labels.append(wrd_score[0])
#         arr = np.append(arr, np.array([wrd_vector]), axis=0)
        
#     # find tsne coords for 2 dimensions
#     tsne = TSNE(n_components=2, random_state=0)
#     np.set_printoptions(suppress=True)
#     Y = tsne.fit_transform(arr)

#     x_coords = Y[:, 0]
#     y_coords = Y[:, 1]
#     # display scatter plot
#     plt.scatter(x_coords, y_coords)

#     for label, x, y in zip(word_labels, x_coords, y_coords):
#         plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
#     plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
#     plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
#     plt.show()
# display_closestwords_tsnescatterplot(model, 'hektaş')
