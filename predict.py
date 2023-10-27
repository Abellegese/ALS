import click
import pickle
from dataset import Dataset
import numpy as np
import pandas as pd

@click.command()
@click.option("--path", default='.', help="path for pickle")
@click.option("--n", default=10, help="dummy user")
@click.option("--fact", default=0.9, help="factor to reduce bias influence")

def predict(path, n, fact):
    #get mapper to map indices to item system id
    mapper =  _get_item_dict()
    #Extract the trait vectors and bias from the pickle
    um, vn, bn = _extract(path)
    #compute the cost
    score = np.inner(um[n], vn) + fact*bn
    #get the index of the top ranked item
    idx = np.argsort(score[-50:][::-1])
    #convert them to system id
    item_id = [mapper[i] for i in idx]
    movies = pd.read_csv('data/25m/movies.csv')
    #search the movies in the csv file
    movies = movies[movies['movieId'].isin(item_id)]
    print(movies[['title','genres']])

#Helper functions
def _get_item_dict():
    data = Dataset('data/25m/ratings.csv', extens='.csv')
    #split the data
    datas = data.split()
    return data.process(data.data, ts=0.8)[3]

def _extract(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['um'], data['vn'], data['bn']

if __name__ == "__main__":
    predict()
