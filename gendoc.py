import os, sys
import glob
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

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
      new_str = d.lower()
      new_str = new_str.replace("\d", "")
      tokenizer = RegexpTokenizer(r'\w+')
      stopword_set = set(stopwords.words('english'))
      dlist = tokenizer.tokenize(new_str)
      dlist = list(set(dlist).difference(stopword_set))
      new_data.extend(dlist)
#       print(dlist)
#       print("-------------------------")
   return new_data

def vectorize(doc, crud_vector):
    vector = []
    crud = [vec[0] for vec in crud_vector]
    for d in doc:
        if d in crud:
            vector

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
print(only_text)
# bag_of_words = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False).fit_transform(splited_labels_from_corpus)
# vect = CountVectorizer(analyzer='word')
# vect_representation= map(vect.fit_transform,clean_text)


# Initialize a CountVectorizer object: count_vectorizer
# vect = CountVectorizer(stop_words="english", analyzer='word', 
#                             ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None)
vect = CountVectorizer(max_features = None)

# Transforms the data into a bag of words
# count_train = count_vec.fit(clean_text)
# bag_of_words = count_vec.transform(clean_text)

X = vect.fit_transform(clean_text) 
# Convert sparse csr matrix to dense format and allow columns to contain the array mapping from feature integer indices to feature names.

count_vect_df = pd.DataFrame(X.todense(),index = all_filenames, columns=vect.get_feature_names())
# Concatenate the original df and the count_vect_df columnwise.

export_csv = count_vect_df.to_csv(foldername+'/text.csv') #Don't forget to add '.csv' at the end of the path

# pd.concat([all_filenames, count_vect_df], axis=1)

if not args.basedims:
    print("Using full vocabulary.")
else:
    print("Using only top {} terms by raw count.".format(args.basedims))
    crud_vector = crud_vector.most_common(args.basedims)

indi_vector = []
if args.tfidf:
    print("Applying tf-idf to raw counts.")
    for doc in data_crude:
        indi_vector  = vectorize(doc, crud_vector)
        tfidf  = TfidfVectorizer()
        crud_vector = tfidf.fit_transform(indi_vector)

if args.svddims:
    print("Truncating matrix to {} dimensions via singular value decomposition.".format(args.svddims))

# THERE ARE SOME ERROR CONDITIONS YOU MAY HAVE TO HANDLE WITH CONTRADICTORY
# PARAMETERS.

print("Writing matrix to {}.".format(args.outputfile))
