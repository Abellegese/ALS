import pandas as pd
import numpy as np
import pickle
class Dataset:
    """
    Dataset Process Pipeline for MovieLense Dataset
    ===============================================
    Args: path to the dataset

    Returns: The proccessed data
    """
    def __init__(self, path, extens='.data'):
        self.path = path
        if extens == '.data':
            self.data = pd.read_csv(path,
                                    header=None,
                                    delimiter='\t',
                                    names=['userId','movieId','ratings','timestamp']
                                   ).drop(columns='timestamp')
        else:
            self.data = pd.read_csv(path).drop(columns='timestamp')

        self.data = self.data.to_numpy()
        

    def split(self, ts:float=0.2):
        """
        Split the data to train and test
        Args: 
            ts: test size
            dtr: data for training
            dts: data for test
        Returns:
            training and test data for both user and item
        """
        data = self.process(self.data, ts)
        user_train, item_train, user_test = data[4], data[5], data[6]

        return user_train, item_train, user_test

    def process(self, data, ts):
        #Shuffle the data
        np.random.shuffle(data)
        #split point
        n = int((1-ts) * len(data))
        #intialize all the data structure
        user_dict = {}
        idx_to_user = []
        item_dict = {}
        idx_to_item = []

        # First build the mappings.
        for idx in range(len(data)):

            user_id = data[idx][0]
            item_id = data[idx][1]

            #Take care of the user data structure
            if user_id not in user_dict:
                idx_to_user.append(int(user_id))
                user_dict[int(user_id)] = len(user_dict)
            
            #Take care of the movie data structure
            if item_id not in item_dict:
                idx_to_item.append(int(item_id))
                item_dict[int(item_id)]=len(item_dict)
        
        #Initialize with empty list all the *trainings* data.
        user_train = [[] for i in range(len(idx_to_user))]
        item_train = [[] for i in range(len(idx_to_item))]

        #Initialize with empty list all the *test* data.
        user_test = [[] for i in range(len(idx_to_user))]
        item_test = [[] for i in range(len(idx_to_item))]

        #create all the data structure using in a loop
        for idx in range(len(data)):
            
            user_id = data[idx][0]
            item_id = data[idx][1]
            rating  = data[idx][2]
            user_idx = user_dict[user_id]
            item_idx = item_dict[item_id]

            if idx < n:
                # Insert into the sparse user and item *training* matrices.
                user_train[user_idx].append((item_idx, float(rating)))
                item_train[item_idx].append((user_idx, float(rating)))

            else:
                # Insert into the sparse user and item *test* matrices.
                user_test[user_idx].append((item_idx, float(rating)))
                item_test[item_idx].append((user_idx,float(rating)))

        return (user_dict,
                idx_to_user,
                item_dict,
                idx_to_item,
                user_train,
                item_train,
                user_test,
                item_test
               )

    def creat_item_feature_dict(self, item_dict):
        """Function to create a data structure for items and its feature"""
        item_feature_dic = {}
        for key in item_dict.keys():
            i = item_dict[key]
            movies = pd.read_csv('data/movies.csv')
            movies = movies[movies['movieId'] == key]
            movies = movies['genres'].str.split('|')
            movies.to_numpy()[0]
            item_feature_dic[i] = (len(movies.to_numpy()[0]), movies.to_numpy()[0])
        with open('item_feature_dict.pkl', 'wb') as f:
            pickle.dump(item_feature_dic, f)

    def creat_feature_item_dict(self, contains_feature=True):
        """ Number of movies for a given genre """
        struct = {}
        for l in FEATURE.values():
            movie = pd.read_csv('data/movies.csv')
            # genre = FEATURE[l]
            items = movie.loc[movie['genres'].str.contains(l)] if contains_feature else movie.loc[~movie['genres'].str.contains(l)]
            items = items['movieId']
            idx = []
            for item in items:
                if item in item_dict.keys():
                    idx.append(item_dict[item])
            struct[l] = idx
        with open(f'test_struc_{contains_feature}.pkl', 'wb') as f:
            pickle.dump(struct, f)
        return struct
