#lexicon = {word: frequency for word, frequency in lexicon2.lexicon.items() if word in model.wv.vocab and frequency>100}
#randomly_chosen_words = random.sample(list(lexicon), 20)
import numpy as np
#x = np.random.randn(500)
#data = [go.Histogram(x=x)]


#py.iplot(data, filename='basic histogram')
print("a kind of cluster similarity score report")
print("-----------------------------------------\n")
allResults=""
curDict=getGroupExcelContents("dataRes.xlsx")
for key in curDict.keys():
    model = FastText.load_fasttext_format(word_embeddings_file) # add the path to the file, applicable to your environment
    allResults+="yap_"+str(key)+"\n"+str(print_joint_similarities(curDict[key],"yap_"+str(key)))
    model = FastText.load_fasttext_format(word_embeddings_fileHewiki2017)
    allResults+="simple_hewiki_2017_"+str(key)+"\n"+str(print_joint_similarities(curDict[key],"simple_hewiki_2017_"+str(key)))
f=open("resultsSimilarGroups.txt","w+")
f.write(allResults)
f.close()
'''
print_joint_similarities(hebrew_disease_words, "hebrew disease words")
print_joint_similarities(hebrew_color_words, "hebrew color words")
print_joint_similarities(sport_words, "sport words")
# Marina------------------
#get_joint_similarities(hebrew_color_words);
#print_joint_similarities(randomly_chosen_words, "randomly chosen hebrew words", print_words=True)
model = FastText.load_fasttext_format(word_embeddings_fileHewiki2017)
print_joint_similarities(hebrew_disease_words, "ahebrew disease words")
print_joint_similarities(hebrew_color_words, "ahebrew color words")
print_joint_similarities(sport_words, "asport words")
'''
