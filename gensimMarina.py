
import openpyxl
import numpy
import itertools
import random
from gensim.models import FastText



import matplotlib.pyplot as plot



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

def getGroupExcelContents(filename): #this will load the first 2 columns from the first row until last one
    book = openpyxl.load_workbook(filename)
    activeSheet=book.active
    cells = activeSheet['A1':'B5150']
    dicOfArrays={}

    for c1,c2 in cells: #c2 will be the values , c1 will be the headers
        curVal=c2.value
        curHeader=c1.value
        if (curVal is not None and curHeader is not None):
            if len(curVal.split()) != 1:
                if (curHeader not in dicOfArrays):
                    dicOfArrays[curHeader]=[]
                dicOfArrays[curHeader].append(curVal)
    return dicOfArrays

def getLexiconFromExcel(filename): #this will load the first 2 columns from the first row until the 1216th one
    book = openpyxl.load_workbook(filename)
    activeSheet=book.active
    cells = activeSheet['A1':'B5150']
    lexicon=[]

    for c1,c2 in cells: #c2 will be the values , c1 will be the headers
        curVal=c2.value
        curHeader=c1.value
        if (curVal is not None and curHeader is not None):
            if len(curVal.split()) == 1:
              if (curVal not in lexicon):
                lexicon.append(curVal)
    return lexicon



word_embeddings_file = 'C:\\OpenUniv\\model.bin'
word_embeddings_fileHewiki2017 = 'C:\\OpenUniv\\hewiki2017FastText.bin'

def similarity_normalize(cosine_distance):
 return (cosine_distance+1)/2

def printHist( group_name, distances):

    plot.title(group_name)
    plot.xlabel("Similarity")
    plot.ylabel("Frequency")
    plot.hist(distances,bins='auto')
    figure = plot.gcf()
    figure.savefig("eval/" +str(group_name)+".png")
    plot.show()

def normalized_similarity(word1, word2, model):
 return similarity_normalize(model.wv.similarity(word1, word2))

#init data

similaritySumYap = 0
varianceSumYap = 0
similaritySumSimpleHewiki = 0
varianceSumSimpleHewiki = 0
isYap = True
def get_joint_similarities(words, group_name, model, verbose=False):
 distances = []
 for pair in itertools.combinations(words, 2):
   distance = normalized_similarity(pair[0], pair[1], model)
   if verbose:
    print("Similarity according to loaded model: {0:.2f} {1} {2}".format(distance, pair[0], pair[1]))
   distances.append(distance)
 printHist( group_name, distances)
 meanVal=numpy.mean(distances)
 #check if it is not nan - is valid data
 global isYap
 global similaritySumSimpleHewiki
 global similaritySumYap
 global varianceSumYap
 global varianceSumSimpleHewiki
 meanVal = numpy.mean(distances)
 varVal = numpy.var(distances)
 if(meanVal<100000):
     if( varVal <100000):
      groupNamesArr.append(group_name)

      if(isYap):
        similarityGroupsYap.append(meanVal)
        similaritySumYap += meanVal
     else:
         similarityGroupsSimpleHewiki.append(meanVal)
         similaritySumSimpleHewiki += meanVal

     if(isYap):
       varianceGroupsYap.append(varVal)
       varianceSumYap += varVal
     else:
       varianceGroupsSimpleHewiki.append(varVal)
       varianceSumSimpleHewiki += varVal

 return numpy.mean(distances), numpy.var(distances)

def print_joint_similarities(words, group_name, model, print_words=False):
    print()
    print(group_name)
    if print_words:
        print(words)
    tmpVar=get_joint_similarities(words, group_name, model)
    print(tmpVar)
    print()
    return tmpVar



lexicon = getLexiconFromExcel("dataResult.xlsx")
length=len(lexicon)-1


def getRandomWords(num):
    rand_chosen_words =[]
    random_index = random.sample(range(1, length-1), num)
    print("random index" +str(random_index))
    for x in range(0, num):
     rIndex=random_index[x]
     rVal=lexicon[rIndex]
     rand_chosen_words.append(rVal)
    return rand_chosen_words

def printAverages(name):
    res= ""
    res+= "Results " + name + " table and average values:\n"

    res+="Groups="+ str(groupNamesArr)

    if isYap:
      res += "\n Yap Similarity of groups=" + str(similarityGroupsYap)
      res += "\nYap Variance of groups="+str(varianceGroupsYap)
      print("Yap Similarity Total Sum " + str(similaritySumYap))
      print("Yap Variance Total Sum " + str(varianceSumYap))
      res+="\nAverage similarity for all groups in Yap =" + str(similaritySumYap/similarityGroupsYap.__len__())
      res+="\nAverage mean for all groups in Yap =" + str(varianceSumYap/varianceGroupsYap.__len__())
    else:
       res += "\n Simple Hewiki Similarity of groups="
       res += str(similarityGroupsSimpleHewiki)
       res += "\nVariance of groups in Simple Hewiki="+str(varianceGroupsSimpleHewiki)
       print(str(similaritySumSimpleHewiki))
       print(str(varianceSumSimpleHewiki))
       res += "\nAverage similarity for all groups in Simple Hewiki =" + str(similaritySumSimpleHewiki/similarityGroupsSimpleHewiki.__len__())
       res += "\nAverage mean for all groups in Simple Hewiki =" + str(varianceSumSimpleHewiki/varianceGroupsSimpleHewiki.__len__())

    print(res)
    return res

#import evaluator
#e = evaluator.Evaluator("datasets/basic")
# AG - evaluation
#hebModel = gensim.models.Word2Vec.load_word2vec_format("model.vec", binary=False)

modelYap = FastText.load_fasttext_format(word_embeddings_file) # add the path to the file, applicable to your environment
modelHewiki2017 = FastText.load_fasttext_format(word_embeddings_fileHewiki2017) # add the path to the file, applicable to your environment

pairWeek1 =['שבוע',
          'שעה']

pairWeek2 =['שבוע',
         'שנה']

pairWeek3 =['שבוע',
         'דקה']

pairWeek4 =['שבוע',
         'חודש']

#pairWeekDistractors1 =[ 'שבוע',
#          'סוף']

pairWeekRandom =['שבוע',  'פרח']

similarityGroupsYap = []
varianceGroupsYap = []
similaritySumYap =0
similaritySumSimpleHewiki =0
varianceSumSimpleHewiki=0
varianceSumYap=0;

similarityGroupsSimpleHewiki = []
varianceGroupsSimpleHewiki = []
groupNamesArr = []

allResults = ""
allResults+=str(print_joint_similarities(pairWeek1,"pairWeek1 Yap", modelYap ))
allResults+=str(print_joint_similarities(pairWeek1,"pairWeek1 Hewiki", modelHewiki2017 ))

allResults+=str(print_joint_similarities(pairWeek2,"pairWeek2 Yap", modelYap ))
allResults+=str(print_joint_similarities(pairWeek2,"pairWeek2 Hewiki", modelHewiki2017 ))

allResults+=str(print_joint_similarities(pairWeek3,"pairWeek3 Yap", modelYap ))
allResults+=str(print_joint_similarities(pairWeek3,"pairWeek3 Hewiki", modelHewiki2017 ))

allResults+=str(print_joint_similarities(pairWeek4,"pairWeek4 Yap", modelYap ))
allResults+=str(print_joint_similarities(pairWeek4,"pairWeek4 Hewiki", modelHewiki2017 ))

#allResults+=str(print_joint_similarities(pairWeekDistractors1,"pairWeekDistractors1 Yap", modelYap ))
#allResults+=str(print_joint_similarities(pairWeekDistractors1,"pairWeekDistractors1 Hewiki", modelHewiki2017 ))

allResults+=str(print_joint_similarities(pairWeekRandom,"pairWeekRandom1 Yap", modelYap ))
allResults+=str(print_joint_similarities(pairWeekRandom,"pairWeekRandom1 Hewiki", modelHewiki2017 ))
print(allResults)



randomly_chosen_words100 = getRandomWords(100)
randomly_chosen_words200 = getRandomWords(200)
randomly_chosen_words300 = getRandomWords(300)
randomly_chosen_words500 = getRandomWords(500)
randomly_chosen_words1000 = getRandomWords(1000)
print("random words similarity score report")
print(randomly_chosen_words100)
allResults += "random words  similarity score report\n"
allResults += str(randomly_chosen_words100)

modelYap = FastText.load_fasttext_format(word_embeddings_file) # add the path to the file, applicable to your environment
modelHewiki2017 = FastText.load_fasttext_format(word_embeddings_fileHewiki2017) # add the path to the file, applicable to your environment

#Second analysys

#hebModel = gensim.models.Word2Vec.load_word2vec_format("model.vec", binary=False)


#hebModel=modelYap
#secondAnalysisRes= e.get_score(hebModel, lambda comp: comp.set_name == 'nn', print_oov=False)
#print("yap goldberg model results")
#print(secondAnalysisRes)

allResults+=str(print_joint_similarities(randomly_chosen_words100,"yap_random_100_words", modelYap ))
allResults+=str(print_joint_similarities(randomly_chosen_words100,"simple_hewiki_2017_random_100words", modelHewiki2017 ))
allResults+=str(print_joint_similarities(randomly_chosen_words200,"yap_random_200_words", modelYap ))
allResults+=str(print_joint_similarities(randomly_chosen_words200,"simple_hewiki_2017_random_200words", modelHewiki2017 ))
allResults+=str(print_joint_similarities(randomly_chosen_words300,"yap_random_300_words", modelYap ))
allResults+=str(print_joint_similarities(randomly_chosen_words300,"simple_hewiki_2017_random_300words", modelHewiki2017 ))
allResults+=str(print_joint_similarities(randomly_chosen_words500,"yap_random_500_words", modelYap ))
allResults+=str(print_joint_similarities(randomly_chosen_words500,"simple_hewiki_2017_random_500words", modelHewiki2017 ))
allResults+=str(print_joint_similarities(randomly_chosen_words1000,"yap_random_1000_words", modelYap ))
allResults+=str(print_joint_similarities(randomly_chosen_words1000,"simple_hewiki_2017_random_1000words", modelHewiki2017 ))
allResults+=str(printAverages("random words"))

similarityGroupsYap = []
varianceGroupsYap = []
groupNamesArr = []
similaritySumYap = 0
varianceSumYap = 0
similarityGroupsSimpleHewiki = []
varianceGroupsSimpleHewiki = []
groupNamesArrSimpleHewiki = []
similaritySumSimpleHewiki = 0
varianceSumSimpleHewiki = 0

print("a kind of cluster similarity score report")
print("-----------------------------------------\n")

curDict=getGroupExcelContents("dataResult.xlsx")
for key in curDict.keys():
    print(str(curDict[key]))
    allResults += "yap_"+str(key)+"\n"+str(print_joint_similarities(curDict[key],"yap_"+str(key), modelYap, True))
    allResults += "simple_hewiki_2017_"+str(key)+"\n"+str(print_joint_similarities(curDict[key],"simple_hewiki_2017_"+str(key),
                                                                                 modelHewiki2017, True))
f=open("eval/resultsSimilarGroups.txt", "w+")

allResults += str(printAverages("Words from similar groups"))

f.write(allResults)
f.close()

