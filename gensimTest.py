{


 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Example for using the gensim library for working with word embeddings\n",
    "make sure to install gensim from the instructions on its website first, so it is availabe to python"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None, # Marina null,
   "metadata": {
    "collapsed": True #Marina true
   },
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt        \n",
    "%matplotlib inline \n",
    "import math"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None, #Marina null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# this will take a minute or two to execute, depending on the speed of your disk and CPU,\n",
    "# it's a very large file for gensim to load and process, but is perfectly normal.\n",
    "# approx. 2GB of memory will be consumed by gensim when it is fully loaded.\n",
    "\n",
    "#word_embeddings_file = '/ssd480/repos/maccabi-nlp-course/wiki.he.bin'\n",
    "word_embeddings_file = 'c:/OpenUniv/Jupiter/model.bin'\n",
    "word_embeddings_fileNotYap = 'c:/OpenUniv/Jupiter/wiki.he.bin'\n",
    "from gensim.models import FastText\n",
    "model = FastText.load_fasttext_format(word_embeddings_file) # add the path to the file, applicable to your environment"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "import itertools\n",
    "import statistics\n",
    "import numpy\n",
    "import random\n",
    "\n",
    "\n",
    "def similarity_normalize(cosine_distance):\n",
    "    return (cosine_distance+1)/2\n",
    "\n",
    "def normalized_similarity(word1, word2):\n",
    "    return similarity_normalize(model.wv.similarity(word1, word2))\n",
    "\n",
    "def vocab_warn(words, embedding_train_corpus_lexicon):\n",
    "    for word in words:\n",
    "        if not word in model.wv.vocab:\n",
    "            print(\"Warning: word {} is not directly embedded in the model\".format(word))\n",
    "        if embedding_train_corpus_lexicon and embedding_train_corpus_lexicon[word] < 100:\n",
    "            print(\"Warning: word {} had low frequency in the embedding's training corpus\".format(word))\n",
    "\n",
    "def get_joint_similarities(words, verbose=False):\n",
    "    #vocab_warn(words)\n",
    "    distances = []\n",
    "    for pair in itertools.combinations(words, 2):\n",
    "        distance = normalized_similarity(pair[0], pair[1])\n",
    "        if verbose:\n",
    "            print(\"Similarity according to loaded model: {0:.2f} {1} {2}\".format(distance, pair[0], pair[1]))\n",
    "        distances.append(distance)\n",
    "        \n",
    "    return numpy.mean(distances), numpy.var(distances)\n",
    "\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "a kind of cluster similarity score report\n",
      "-----------------------------------------\n",
      "\n",
      "\n",
      "hebrew disease words\n",
      "(0.5466777578639432, 0.002931648354502827)\n",
      "\n",
      "\n",
      "hebrew color words\n",
      "(0.5285438495776661, 0.002001055223761304)\n",
      "\n"
     ]
    },
    {
     "ename": "NameError",
     "evalue": "name 'randomly_chosen_words' is not defined",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-4-b22d6bc993f9>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[1;32m     43\u001b[0m \u001b[0mprint_joint_similarities\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mhebrew_disease_words\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m\"hebrew disease words\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     44\u001b[0m \u001b[0mprint_joint_similarities\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mhebrew_color_words\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m\"hebrew color words\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 45\u001b[0;31m \u001b[0mprint_joint_similarities\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mrandomly_chosen_words\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m\"randomly chosen hebrew words\"\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mprint_words\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mTrue\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m: name 'randomly_chosen_words' is not defined"
     ],
     "output_type": "error"
    }
   ],
   "source": [
    "hebrew_disease_words = [\n",
    "    'צהבת',\n",
    "    'גחלת',\n",
    "    'שפעת',\n",
    "    'שעלת',\n",
    "    'אסטמה',\n",
    "    'רככת',\n",
    "    'שחמת',\n",
    "    'צפדינה',\n",
    "    'סכרת',\n",
    "    'אנמיה',\n",
    "    'מלריה',\n",
    "    'שחפת']\n",
    "\n",
    "hebrew_color_words = [\n",
    "    'לבן',\n",
    "    'צהוב',\n",
    "    'ורוד',\n",
    "    'כחול',\n",
    "    'ירוק',\n",
    "    'סגול',\n",
    "    'שחור',\n",
    "    'כתום',\n",
    "    'אדום',\n",
    "    'חום',\n",
    "    'אפור'\n",
    "]\n",
    "\n",
    "def print_joint_similarities(words, group_name, print_words=False):\n",
    "    print()\n",
    "    print(group_name)\n",
    "    if print_words:\n",
    "        print(words)\n",
    "    print(get_joint_similarities(words))\n",
    "    print()\n",
    "\n",
    "#lexicon = {word: frequency for word, frequency in lexicon2.lexicon.items() if word in model.wv.vocab and frequency>100}\n",
    "#randomly_chosen_words = random.sample(list(lexicon), 20)  \n",
    "    \n",
    "print(\"a kind of cluster similarity score report\")\n",
    "print(\"-----------------------------------------\\n\")\n",
    "\n",
    "print_joint_similarities(hebrew_disease_words, \"hebrew disease words\")\n",
    "print_joint_similarities(hebrew_color_words, \"hebrew color words\")\n",
    "print_joint_similarities(randomly_chosen_words, \"randomly chosen hebrew words\", print_words=True)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None, #null,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.wv.vocab['מריינפלדה']"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
import numpy
import itertools
from gensim.models import FastText
import random

'''
def similarity_normalize(cosine_distance):
 return (cosine_distance+1)/2

def normalized_similarity(word1, word2):
 return similarity_normalize(model.wv.similarity(word1, word2))

def get_joint_similarities(words, verbose=False):
 distances = []
 for pair in itertools.combinations(words, 2):
   distance = normalized_similarity(pair[0], pair[1])
   if verbose:
    print("Similarity according to loaded model: {0:.2f} {1} {2}".format(distance, pair[0], pair[1]))
   distances.append(distance)
 return numpy.mean(distances), numpy.var(distances)

def print_joint_similarities(words, group_name, print_words=False):
 print(),
 print(group_name),
 if print_words:
  print(words)
  print(get_joint_similarities(words))
  print()
'''

#lexicon = {word: frequency for word, frequency in lexicon2.lexicon.items() if word in model.wv.vocab and frequency>100}
#randomly_chosen_words = random.sample(list(lexicon), 20)

word_embeddings_file = 'C:\\OpenUniv\\model.bin'

model = FastText.load_fasttext_format(word_embeddings_file) # add the path to the file, applicable to your environment

hebrew_disease_words = [
    'צהבת',
    'גחלת',
    'שפעת',
    'שעלת',
    'אסטמה',
    'רככת',
    'שחמת',
    'צפדינה',
    'סכרת',
    'אנמיה',
    'מלריה',
    'שחפת']

hebrew_color_words = [
    'לבן',
    'צהוב',
    'ורוד',
    'כחול',
    'ירוק',
    'סגול',
    'שחור',
    'כתום',
    'אדום',
    'חום',
    'אפור'
]
sport_words=[
'קַשׁתוּת',
'זִירָה',
'חֵץ',
'אַתלֵ',
'אַתלֵטִיקָה',
'אקסל',
'נוצית',
'כַּדוּר',
'בסיס',
'בייסבול',
'כדורסל',
'עטל',
'שַׁרבִּיט',
'חמאה',
'להכות',
'ביאתלון'
'אופניים'
'בִּילְיַארד'
'בובסלי',
'בּוּמֵרַנְג',
'קשת',
'בָּאוּלֵר',
'בָּאוּלִינְג',
'מִתאַגרֵף',
'אִגרוּף'
]
def similarity_normalize(cosine_distance):
 return (cosine_distance+1)/2

def normalized_similarity(word1, word2):
 return similarity_normalize(model.wv.similarity(word1, word2))

def get_joint_similarities(words, verbose=False):
 distances = []
 for pair in itertools.combinations(words, 2):
   distance = normalized_similarity(pair[0], pair[1])
   if verbose:
    print("Similarity according to loaded model: {0:.2f} {1} {2}".format(distance, pair[0], pair[1]))
   distances.append(distance)
 return numpy.mean(distances), numpy.var(distances)

def print_joint_similarities(words, group_name, print_words=False):
    print()
    print(group_name)
    if print_words:
        print(words)
    print(get_joint_similarities(words))
    print()

#lexicon = {word: frequency for word, frequency in lexicon2.lexicon.items() if word in model.wv.vocab and frequency>100}
#randomly_chosen_words = random.sample(list(lexicon), 20)
import plotly.plotly as py
import plotly.graph_objs as go

import numpy as np

x = np.random.randn(500)
data = [go.Histogram(x=x)]

py.iplot(data, filename='basic histogram')
print("a kind of cluster similarity score report")
print("-----------------------------------------\n")

print_joint_similarities(hebrew_disease_words, "hebrew disease words")
print_joint_similarities(hebrew_color_words, "hebrew color words")
print_joint_similarities(sport_words, "sport words")
# Marina------------------
get_joint_similarities(hebrew_color_words);
#print_joint_similarities(randomly_chosen_words, "randomly chosen hebrew words", print_words=True)
