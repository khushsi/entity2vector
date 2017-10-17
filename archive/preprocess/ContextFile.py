import nltk
import csv
import sys
import os

from nltk.stem.snowball import SnowballStemmer
# documentsfile = open('/Users/khushsi/Downloads/entity2vector-ir/data/ir_textbook.txt.allconcepts','r')
# wikipediadocs = open('/Users/khushsi/Downloads/concept_extraction/acm_dl/data/keyphrase/textbook/iirmirbook.csv','r')
wikipediadocs = open('/Users/khushsi/Downloads/concept_extraction/acm_dl/data/keyphrase/textbook/final_merged_mir.csv','r')
# ps = nltk.stem.SnowballStemmer()
stemmer = SnowballStemmer("english", ignore_stopwords=True)


IS_STEM=False

def myownstem(word):
    stemword = word
    if IS_STEM:
        stemword = stemmer.stem(word)
    return stemword

def multiwordstem(word_list ):
    for i in range(len(word_list)):
        word_list[i] = myownstem(word_list[i])
    return ' '.join(word_list)

def load_stopwords():
    print (os.getcwd())
    STOPWORD_PATH = '/Users/khushsi/Downloads/concept_extraction/acm_dl/data/stopword/stopword_min.txt'
    dict = set()
    file = open(STOPWORD_PATH, 'r')
    for line in file:
        dict.add(line.lower().strip())
    dict = set()
    return dict

stopwords = load_stopwords()

conceptdocs={}
from nltk.corpus import stopwords

def stem(x):
    try:
        return ps.stem(x)
    except Exception:
        return x


def removeStopword(x):
    xlist = x.split(" ")
    nx = [xi for xi in xlist if xi not in stopwords]
    return ' '.join(nx)

def  lower(x):
    return x.lower()

Max_concepts = 10
## Extract Concepts
CONCEPT_FOLDER_BASE="/Users/khushsi/Downloads/concept_extraction/acm_dl/src/keyphrase_output_199/"
CONCEPTS = ['gmgTFIDF1340', 'gmgTFIDFNPWP1340', 'gold', 'gold_E2v', 'gTFIDFNPWP1340', 'mCopyRNN', 'mgTFIDF1340', 'TFIDF1340']
iCount = 0
l_concepts = set()
for file in os.listdir(CONCEPT_FOLDER_BASE+CONCEPTS):
    if file.endswith("txt"):
        for line in open(file,'r').readlines():
            if(iCount < Max_concepts):

                xline = stem(removeStopword(lower(line).split(",")[0]))
                if (len(xline.strip()) >  2):
                    if(len(xline.split(" ")) < 4):
                        l_concepts.add(xline)
                    else:
                        l_concepts.add(xline.split(" ")[0:3])
                iCount = iCount + 1


import re
def filltokendict(doc):
    tokens = [re.sub(r'\W+|\d+', '', word) for word in doc.split()]
    tokens = list(filter(lambda x: len(x) > 1, tokens))
    tokens = list(map( lower, tokens))
    tokens = list(filter(lambda x: x not in stopwords , tokens))
    tokens = list(map(stem,tokens))
    # print(tokens)

    ngrams = nltk.ngrams(tokens,n=6)
    for ngram in ngrams:
        token=ngram[0]
        # print(ngram[0])

        if token in l_concepts:
            if token in conceptdocs:
                conceptdocs[token] += ngram[1:]
            else:
                conceptdocs[token] = list(ngram[1:])
            # print(conceptdocs[token])

        token=ngram[5]
        if token in l_concepts:
            if token in conceptdocs:
                conceptdocs[token] += ngram[:5]
                # print(ngram[:4])
            else:
                conceptdocs[token] = list(ngram[:5])
            # print(conceptdocs[token])

    ngrams = nltk.ngrams(tokens, n=7)
    for ngram in ngrams:

        token=' '.join([ngram[5],ngram[6]])
        # print(token)
        if token in l_concepts:
            # print(token)
            if token in conceptdocs:
                conceptdocs[token] += ngram[:5]
            else:
                conceptdocs[token] = list(ngram[:5])
            # print(conceptdocs[token])

        token=' '.join([ngram[0],ngram[1]])
        if token in l_concepts:
            # print(token)
            if token in conceptdocs:
                conceptdocs[token] += ngram[2:]
            else:
                conceptdocs[token] = list(ngram[2:])
            # print(conceptdocs[token])

    ngrams = nltk.ngrams(tokens, n=8)
    for ngram in ngrams:

        token=' '.join([ngram[5],ngram[6],ngram[7]])
        # print(token)
        if token in l_concepts:
            # print(token)
            if token in conceptdocs:
                conceptdocs[token] += ngram[:5]
            else:
                conceptdocs[token] = list(ngram[:5])
            # print(conceptdocs[token])

        token=' '.join([ngram[0],ngram[1],ngram[2]])
        if token in l_concepts:
            print(token)
            print(ngram)
            if token in conceptdocs:
                conceptdocs[token] += ngram[3:]
                print("8", ngram[3:])

            else:
                conceptdocs[token] = list(ngram[3:])
            # print(conceptdocs[token])

csvreader = csv.reader(wikipediadocs, delimiter='\t')
csv.field_size_limit(sys.maxsize)

for doc in csvreader:
    filltokendict(doc[1])
print(len(conceptdocs))
# print(len(conceptdocs["inform retriev"]))
print(conceptdocs.keys())

# csvreader = csv.reader(documentsfile, delimiter='\t')
# for doc in csvreader:
#     filltokendict(doc[1])

fw = open("conceptdocs_1908.csv","w")

for key in conceptdocs:
    print(key)
    fw.write(key+"\t"+' '.join(conceptdocs[key])+ "\n")

