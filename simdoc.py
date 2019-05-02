import os, sys
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def find_cosine(vectorfile):
    to_csv = pd.read_csv(vectorfile,index_col=0)
    dicti = to_csv.to_dict('split')
    nlist = []
    columns = dicti['columns']
    index = [i.split('_')[0] for i in dicti['index']]
    data =  dicti['data']
    category={}
    for count,i in enumerate(data):
        nlist.append((index[count],i))
    for vectorfile,vector in nlist:
        if vectorfile  not in category:
            category[vectorfile]=[vector]
        else:
            category[vectorfile]+=[vector]
    # print(category.keys())
    sim = cosine_similarity(list(category.values())[0],list(category.values())[1])
    avg_sim = np.mean(sim)
    print('Average similarity  ', list(category.keys())[0],list(category.keys())[1],' : ', avg_sim)
    sim = cosine_similarity(list(category.values())[0],list(category.values())[0])
    avg_sim = np.mean(sim)
    print('Average similarity ', list(category.keys())[0],list(category.keys())[0],' : ',avg_sim)
    sim = cosine_similarity(list(category.values())[1],list(category.values())[0])
    avg_sim = np.mean(sim)
    print('Average similarity  ', list(category.keys())[1],list(category.keys())[0],' : ',avg_sim)
    sim = cosine_similarity(list(category.values())[1],list(category.values())[1])
    avg_sim = np.mean(sim)
    print('Average similarity  ', list(category.keys())[1],list(category.keys())[1],' : ',avg_sim)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compute some similarity statistics.")
    parser.add_argument("vectorfile", type=str,
                    help="The name of the input  vectorfile for the matrix data.")

    args = parser.parse_args()

    print("Reading matrix from {}.".format(args.vectorfile))

    find_cosine(args.vectorfile)
