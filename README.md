Yap word2vec model comparison to  Simple wiki word2vec model 
Yap model evaluation compares result of fasttext model created on basis of yap morphological preprocessing of Hebrew wikipedia 
versus  fasttext model created without yap morphological preprocessing on same input corpus(Hebrew wikipedia).


Here we provide:

A python implementation of the method
A suite of matching datasets in Hebrew 
Requirements
Python 3.7
PyCharm
gensim (only for the example script)
Example
Run the following file in PyCharm:
gensim.py


The code in gensim.py loads a gensim word2vec Yap model and simple model without Yap and runs evaluation on the different datasets.
Notice that the models it uses (model.vec and simpleHewiki2017.bin) covers hebrew wikipedia from 2017 July.


The provided dataset

dataResult.xlxs contains 5180 words from similar groups.
all experiment results and intermediate data exist in directory eval.
You can see all the experiments results in file eval/outputResultsAll.txt
All the histogram pictures exists in directory eval
results images related to experiments exists in eval directory.
For example: yap_adjectives.png for adjectives similar group for yap model and simple_hewiki_2017_adjectives.png for simple model without yap. 

References
If you make use of this software for research purposes, we'll appreciate citing the following:

@InProceedings{
  author    = {Marina Voloshin, Rony Boter},
  title     = {Improving Reliability of Word Similarity Evaluation by using of Yap preprocessing},
  month     = {August},
  year      = {2018},
  address   = {Tel Aviv Israel}
  document = {NLP 2018 course}
}
