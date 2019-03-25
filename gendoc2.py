import os, sys
import glob
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
import os
from os import listdir
from os.path import isfile, join
from nltk.probability import FreqDist
import zipfile


def load_data(folder):
    dir_list = sorted(os.listdir(folder))
    if '.DS_Store' in dir_list:
        dir_list.remove('.DS_Store')    

    link_list = []
    all_data = []
    for dir in dir_list:
        category_list = []
        category_list.append(dir)
        subdir = os.path.join(folder,dir)
        category_list.append(subdir)
        article_name =[]
        article_data = []
        for file in sorted(os.listdir(subdir)):
            if file.endswith('.txt'):
                file_path = os.path.join(subdir,file)
                article_name.append(dir+'_'+file)
                f = open(file_path,"r")
                text = f.read()
                article_data.append(text)
        category_list.append(article_name)
        
        category_list.append(article_data)
        all_data.append(category_list)
#     print(all_data[0][0],all_data[0][1],all_data[0][2])
#     # folder, folder_path, list of articles, article_Data
    
#     print(all_data[1][0],all_data[1][1],all_data[1][2])
    return all_data
                
             
def nlp_clean(data):
   new_data = []
   for d in data:
      line = d.replace('\n', '')
      line2 = re.sub('[^\sA-Za-z]+', '', line).lower()
      new_data.append(line2)
   return new_data


def wm2df(wm, feat_names, file_names):
    
    # create an index for each row
#     doc_names = ['Doc{:d}'.format(idx) for idx, _ in enumerate(wm)]
   
    return(df)
  



parser = argparse.ArgumentParser(description="Generate term-document matrix.")
parser.add_argument("-T", "--tfidf", action="store_true", help="Apply tf-idf to the matrix.")
parser.add_argument("-S", "--svd", metavar="N", dest="svddims", type=int,
                    default=None,
                    help="Use TruncatedSVD to truncate to N dimensions")
parser.add_argument("-B", "--base-vocab", metavar="M", dest="basedims",
                    type=int, default=None,
                    help="Use the top M dims from the raw counts before further processing")
parser.add_argument("foldername", type=str,
                    help="The base folder name containing the two topic subfolders.")
parser.add_argument("outputfile", type=str,
                    help="The name of the output file for the matrix data.")

args = parser.parse_args()

print("Loading data from directory {}.".format(args.foldername))
foldername = args.foldername 


all_data = load_data(foldername)
only_text = [] 
only_text.extend(all_data[0][3])
only_text.extend(all_data[1][3])
all_filenames = []
all_filenames.extend(all_data[0][2])
all_filenames.extend(all_data[1][2])

clean_text = nlp_clean(only_text)
print(clean_text)
# print(only_text)


if not args.basedims:
    print("Using full vocabulary.")
    # set of documents
    corpora = clean_text

    # instantiate the vectorizer object
    cvec = CountVectorizer(lowercase=False)

    # convert the documents into a document-term matrix
    wm = cvec.fit_transform(corpora)

    # retrieve the terms found in the corpora
    feat_names = cvec.get_feature_names()    
    df = pd.DataFrame(data=wm.toarray(), index=all_filenames,
                      columns=feat_names)
#     # create a dataframe from the matrix
#     df = wm2df(wm, tokens, all_filenames)

else:
    print("Using only top {} terms by raw count.".format(args.basedims))
    crud_vector = crud_vector.most_common(args.basedims)

indi_vector = []
if args.tfidf:
    print("Applying tf-idf to raw counts.")
    tfidf_vectorizer = TfidfVectorizer(strip_accents='ascii',stop_words='english',min_df=0.0005)             
    codebook_vectorizer = tfidf_vectorizer.fit_transform(clean_text).toarray()
    idf = tfidf_vectorizer.idf_ 
    df = pd.DataFrame(data=codebook_vectorizer, index=all_filenames,columns=tfidf_vectorizer.get_feature_names)

if args.svddims:
    print("Truncating matrix to {} dimensions via singular value decomposition.".format(args.svddims))

# THERE ARE SOME ERROR CONDITIONS YOU MAY HAVE TO HANDLE WITH CONTRADICTORY
# PARAMETERS.

print("Writing matrix to {}.".format(args.outputfile))
export_csv = df.to_csv(foldername+'/text.csv') #Don't forget to add '.csv' at the end of the path