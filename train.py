from dataset import Dataset
from model import ALSModel
import pandas as pd
import numpy as np
import click


@click.command()
@click.option("--path", default='.', help="Path for te dataset")
@click.option("--epoch", default=10, help="Path for te dataset")
@click.option("--ts", default=0.1, help="test size")
@click.option("--lamda", default=0.001, help="lamda: hyper param")
@click.option("--gamma", default=0.001, help="gamma: hyper param")
@click.option("--thau", default=0.5, help="thau: hyper param")
@click.option("--dim", default=5, help="dim: hyper param")
@click.option("--feature", default=False, help="Bool to check training with feature")

def train(path, epoch, ts, lamda, gamma, thau, dim, feature):
    # create Dataset pipeline instance
    data = Dataset(path, extens='.csv')
    #split the data
    datas = data.split()
    #get some datastructures
    returns = data.process(data.data, ts=0.1)
    idx_to_item, item_dict = returns[3], returns[2]
    user_train, item_train, user_test = datas
    model = ALSModel(
                    data=(user_train, item_train), 
                    test_data=user_test,
                    mapper=idx_to_item, 
                    item_dict=item_dict, 
                    lamda=lamda, 
                    gamma=gamma, 
                    thau=thau, 
                    latent_dim=dim,
                    with_feature=feature
                    )

    model.fit(epoch)

if __name__ == '__main__':
    #python3 train.py --path 'data/ratings_small.csv' --epoch=10 --ts 0.1
    train()