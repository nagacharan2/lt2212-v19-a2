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
      dlist = " ".join(dlist)
      new_data.append(dlist)

   return new_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate term-document matrix.")
    parser.add_argument("-T", "--tfidf", action="store_true",help="Apply tf-idf to the matrix.")
    parser.add_argument("-S", "--svd", metavar="N", dest="svddims", type=int,
                        default=0,
                        help="Use TruncatedSVD to truncate to N dimensions")
    parser.add_argument("-B", "--base-vocab", metavar="M", dest="basedims",
                        type=int, default=0,
                        help="Use the top M dims from the raw counts before further processing")
    parser.add_argument("foldername", type=str, #default='/Users/nagacharan/Documents/NLP/Statistical Methods/lt2212-v19-a2/reuters-topics',
                        help="The base folder name containing the two topic subfolders.")
    parser.add_argument("outputfile", type=str, #default="new.txt",
                        help="The name of the output file for the matrix data.")

    args = parser.parse_args()

    print("Loading data from directory {}.".format(args.foldername))
    foldername = args.foldername 
    filename = args.outputfile

    all_data = load_data(foldername)
    only_text = [] 
    only_text.extend(all_data[0][3])
    only_text.extend(all_data[1][3])

    all_filenames = []
    all_filenames.extend(all_data[0][2])
    all_filenames.extend(all_data[1][2])
                      
    clean_text = nlp_clean(only_text)

    if not args.basedims:
        print("Using full vocabulary.")
        vect = CountVectorizer(max_features = None)
    else:
        print("Using only top {} terms by raw count.".format(args.basedims))
        vect = CountVectorizer(max_features = args.basedims)

    X = vect.fit_transform(clean_text) 
    count_vect_df = pd.DataFrame(X.todense(),index = all_filenames, columns=vect.get_feature_names())
    export_csv = count_vect_df.to_csv(foldername+'/'+filename +'_vector.csv') #Don't forget to add '.csv' at the end of the path

    if args.tfidf:
        print("Applying tf-idf to raw counts.")
        if args.basedims:
            tfidvect = TfidfVectorizer(max_features = args.basedims)
        else:
            tfidvect = TfidfVectorizer(max_features = None)
        Y = tfidvect.fit_transform(clean_text) 
        tfid_vect_df = pd.DataFrame(Y.todense(),index = all_filenames, columns=tfidvect.get_feature_names())
        export_csv = tfid_vect_df.to_csv(foldername+ '/'+filename +'_tfidf_vector.csv') #Don't forget to add '.csv' at the end of the path

    if args.svddims:
        print("Truncating matrix to {} dimensions via singular value decomposition.".format(args.svddims))
        if args.svddims < len(vect.get_feature_names()):
            svdvect = TruncatedSVD(n_components = args.svddims)
            if args.tfidf:
                Z = svdvect.fit_transform(Y)
                svd_tf_df = pd.DataFrame(Z,index = all_filenames, columns =[i for i in  range(0,args.svddims)])
                export_csv = svd_tf_df.to_csv(foldername+'/'+filename +'_tfidf_svd.csv') #Don't forget to add '.csv' at the end of the path
            U = svdvect.fit_transform(X)
            svd_df = pd.DataFrame(U,index = all_filenames, columns =[i for i in  range(0,args.svddims)])
            export_csv = svd_df.to_csv(foldername+'/'+filename +'_vector_svd.csv') #Don't forget to add '.csv' at the end of the path


    print("Writing matrix to {}.".format(args.outputfile))
