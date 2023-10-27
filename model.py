import pandas as pd
import numpy as np
from numpy.linalg import norm, solve
import time
from tqdm import tqdm as tq
import pickle
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import scipy.linalg as sla
import matplotlib.pyplot as plt
from concurrent.futures import wait
from utils import Utils


class ALSModel(Utils):
    """
    Alternate Least Square Implementation for Recommendation System
    ===============================================================
    Args: data: spasre data for both item and user
          test_data: splited user data form the original sparse
    Returns : ALl the params intialized in the class constructor         
    """

    def __init__(self, data, 
                       test_data,
                       mapper,
                       item_dict,
                       with_feature=False,
                       latent_dim=5, 
                       thau = 0.001,
                       gamma=0.001,
                       lamda=0.001
                ):

        self.user_data, self.item_data = data
        self.test_data = test_data
        self.num_user = len(self.user_data)
        #Mapper
        self.mapper = mapper
        self.item_dict = item_dict
        #Bool to check training with feature
        self.with_feature = with_feature
        #Hyperparams
        self.gamma = gamma
        self.lamda = lamda
        self.thau = thau
        self.num_item = len(self.item_data)
        self.latent_dim = latent_dim
        self.thau = thau
        #Trait vector initialization
        self.user_matrix = np.random.normal(0, 1/np.sqrt(latent_dim),size=(self.num_user, self.latent_dim))
        self.item_matrix = np.random.normal(0, 1/np.sqrt(latent_dim),size=(self.num_item, self.latent_dim))
        #Bias initialization
        self.user_bias = np.zeros((self.num_user))
        self.item_bias = np.zeros((self.num_item))
        #ThauI
        self.thauI = thau*np.eye(latent_dim)
        #cost
        self.cost = []
        self.rmse = []
        #cost test
        self.cost_test = []
        self.rmse_test = []
        #Initialize Item-feature & Feature-Item datastructure
        self.item_feature_dict = self.get_features()
        #Select item for a given feature
        self.feature_item_dict_true = self.get_items()
        #Select item that does not belong to a given feature
        self.feature_item_dict_false = self.get_items(contains_feature=False)
        self.FEATURE = self._get_features_dict()
        #Feature Initialization
        self.feature = np.zeros((len(self.FEATURE), latent_dim))
        #Defining Queue to schedule order execution of the functions in distributed process
        #because item_update function depends on the values of user_item functions
        self.queue = multiprocessing.Queue()

    def _update_user_parellel(self, range):
        # update user bias + self.self.user_matrix for all users
        for m in range:
            bm = 0
            count = 0
            for n, r in self.user_data[m]:
                inner = np.inner(self.user_matrix[m, :], self.item_matrix[n, :])
                bm += self.lamda*(r - (inner + self.item_bias[n]))
                count += 1
            self.user_bias[m] =  bm / (self.lamda * count + self.gamma)

            # Set the trait vectors
            b = np.zeros(self.latent_dim)
            outer = 0
            for n, r in self.user_data[m]:
                b += self.lamda *self.item_matrix[n, :]*(r - (self.user_bias[m] + self.item_bias[n]))                
                outer += self.lamda*(np.outer(self.item_matrix[n, :], self.item_matrix[n, :]))
            outer += self.thauI

            self.user_matrix[m, :] = np.linalg.inv(outer) @ b

    def _update_item_parallel(self, range):
        # update user bias + self.self.user_matrix for all users
        for m in range:
            bn = 0
            count = 0
            for n, r in self.item_data[m]:
                inner = np.inner(self.item_matrix[m, :], self.user_matrix[n, :])
                bn += self.lamda*(r - (inner + self.user_bias[n]))
                count += 1
            self.item_bias[m] =  bn / (self.lamda * count + self.gamma)

            # Set the trait vectors
            b = np.zeros(self.latent_dim)
            outer = 0
            for n, r in self.item_data[m]:
                b += self.lamda *self.user_matrix[n, :] *(r - (self.user_bias[n] + self.item_bias[m]))                
                outer += self.lamda*(np.outer(self.user_matrix[n, :], self.user_matrix[n, :]))
            #Update for feature
            if self.with_feature:
                f = 0
                Fn, l = self.item_feature_dict[m]
                for feat in  l:
                    idx = list(self.FEATURE.values()).index(feat)
                    f += self.feature[idx, :]
                #Normilize
                f = f / Fn 
                outer += self.thauI
                self.item_matrix[m, :] = np.linalg.inv(outer) @ (b + f)
            else:
                outer += self.thauI
                self.item_matrix[m, :] = np.linalg.inv(outer) @ b
                
    def _update_feature_parallel(self, range):
        #Update it for each features
        for m in range:
            #Initialize Accumulators   
            v, f, F = 0, 0, 1
            #Update for the movie that belongs to a given feature
            for n in self.feature_item_dict_true[m]:
                v += self.item_matrix[n, :]
                Fn, l = self.item_feature_dict[n]
                v = v / np.sqrt(Fn)
            #Update for the movie that does not belong to a given feature
            for n in self.feature_item_dict_false[m]:
                Fn, l = self.item_feature_dict[n]
                for feat in  l:
                    idx = list(self.FEATURE.values()).index(feat)
                    f += self.feature[idx, :]
                #Compute normilizer
                F += Fn

            self.feature[m, :] = (v - f) / F
            
    def fit(self,epoch=10, file_name='docs/metrics', plot=True):
        
        for _ in tq(range(epoch)):
            #Paralleilze the training
            with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                #Execute training in parallel for all users
                job1 = executor.submit(self._update_user_parellel, [i for i in range(self.num_user)])
                #Execute training in parallel for all items
                job2 = executor.submit(self._update_item_parallel, [i for i in range(self.num_item)])
                if self.with_feature:
                    #Execute training in parallel for all features
                    job3 = executor.submit(self._update_feature_parallel, [i for i in range(len(self.FEATURE))])
            #train metrics
            self.cost.append(self.compute_cost(data=self.user_data)[0])
            self.rmse.append(self.compute_cost(data=self.user_data)[1])
            # test metrics
            self.cost_test.append(self.compute_cost(data=self.test_data)[0])
            self.rmse_test.append(self.compute_cost(data=self.test_data)[1])

        #plot the error
        self.plot(file_name, plot=plot)
        # Store the model metrics and params
        self.pickle()
    
    def predict(self, n, mapper:list, fact:float=1.0):
        # n = dummy user
        # mapper: item idx to system id
        score = np.inner(self.user_matrix[n], self.item_matrix) + fact*self.item_bias
        idx = np.argsort(score[-50:][::-1])
        item_id = [mapper[i] for i in idx]
        movies = pd.read_csv('data/movies.csv')
        movies = movies[movies['movieId'].isin(item_id)]
        return  movies[['title','genres']] # top 50
        
