import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd

class Utils:
    """Util calss contains extra helper functions for the main model class"""

    def plot(self, file_name:str, plot:bool, save=True):
        #Lets only plot RMSE..[Useful metrics]
        fig, ax = plt.subplots(figsize=(14,6))
        plt.subplot(1, 2, 1)  # row 1, column 2, count 1
        plt.plot(np.arange(0, len(self.rmse),1), np.array(self.rmse), color='g')
        plt.plot(np.arange(0, len(self.rmse_test),1), np.array(self.rmse_test), color='r')
        plt.legend(['training rmse', 'test rmse'])
        plt.xlabel('# iterations')
        plt.ylabel('RMSE')
        #plot for log loss
        if plot:
            plt.subplot(1, 2, 2)  # row 1, column 2, count 1
            plt.plot(np.arange(0, len(self.cost),1), np.array(self.cost), color='g')
            plt.legend(['training loss'])
            plt.xlabel('# iterations')
            plt.ylabel('Log Loss')
        if save:
            plt.savefig(f'{file_name}.pdf')
        plt.show()

    def plot_power(self):

        unique_user, unique_item, user_count, item_count = self.frequency()

        fig, ax = plt.subplots()
        #ax.scatter(unique_user, user_count, label = "user", marker = "o", s=5)
        ax.fill_between(item_count, unique_item, color='blue')
        # ax.set_xscale("log")
        # ax.set_yscale("log")
        ax.set_xlabel("product", fontsize=12)
        ax.set_ylabel("popularity", fontsize=14)
        ax.set_title("power law distribution", fontsize=12)

        ax.legend(loc='upper right', fontsize=12)

        plt.show()
    
    def frequency(self):
        """
        Function to compute the frequency of the item and user
        used to plot power law graphs
        """
        userIdx = [len(row) for row in self.user_data] 
        itemIdx = [len(row) for row in self.item_data] 

        unique_user = np.unique(userIdx)
        unique_item = np.unique(itemIdx)

        item_count = np.zeros(len(unique_item))
        user_count = np.zeros(len(unique_user))

        for item in itemIdx:
            index = unique_item.searchsorted(item)
            item_count[index] += 1

        for item in userIdx:
            index = unique_user.searchsorted(item)
            user_count[index] += 1

        return (unique_user, unique_item, user_count, item_count)
        
    @staticmethod
    def plot_rating_dist():
        #getting rating data
        ratings = pd.read_csv('data/ratings_small.csv')['rating']
        unique, count = np.unique(ratings.to_numpy().astype('int'), return_counts=True)
        plt.bar(unique, count)
        plt.xlabel('frequency')
        plt.ylabel('ratings')

    def pickle(self):

        """Serializing the trained params and metrics"""
        params = {
                  "um": self.user_matrix , 
                  "vn": self.item_matrix, 
                  "bm": self.user_bias,
                  "bn": self.item_bias,
                  "feature": self.feature,
                  "cost": self.cost,
                  "rmses": self.rmse,
                  "cost_test": self.cost_test,
                  "rmses_test": self.rmse_test
                  }

        # Open a file in binary write mode.
        with open("parameters/parameters.pickle", "wb") as f:
            # Pickle the object to the file.
            pickle.dump(params, f)
    
    def get_features(self):
        """Function to extract Fn and features for a given movie"""
        #i = self.mapper[i]
        with open('docs/item_feature_dict.pkl', 'rb') as f:
            dicts = pickle.load(f)
        return dicts


    def get_items(self, contains_feature=True):
        """ 
        Number of movies for a given genre 
        Args: contains_feature: Used to return the movies that corresponding to a 
              certain feature if it is True else it returns that the movie that not 
              corresponding to a given feature
        """
        # Extrat item given genres contains in csv file
        if contains_feature:
            with open('docs/feature_item_dict_True.pkl', 'rb') as f:
                dicts = pickle.load(f)
        else:
            with open('docs/feature_item_dict_False.pkl', 'rb') as f:
                dicts = pickle.load(f)
        # System Id for a given generes
        return dicts

    def compute_cost(self, data):
        #data: sparse data for user only
        loss, rmse, count = 0, 0, 0
        
        for i in range(len(data)):
            for n, r in data[i]:
                error = (r - (np.inner(self.user_matrix[i, :], self.item_matrix[n, :]) + self.user_bias[i] + self.item_bias[n]))**2
                loss += (self.lamda / 2) * error
                rmse += error
                count += 1
                
        loss += (self.gamma/2)*(np.dot(self.item_bias, self.item_bias) + (self.gamma/2)*np.dot(self.user_bias, self.user_bias))
        #loss regulizer for features
        if self.with_feature:
            for l in range(len(self.FEATURE)):
                loss += (self.thau/2)*np.dot(self.feature[l, :], self.feature[l, :])
        
        for j in range(self.num_item):
            if self.with_feature:
                #Compute features
                f = 0
                # l: l in features(n)
                Fn, l = self.item_feature_dict[j]
                for feat in  l:
                    idx = list(self.FEATURE.values()).index(feat)
                    f += self.feature[idx, :]
                #Normilize
                f = f / np.sqrt(Fn)
                loss += (self.thau/2)*np.dot((self.item_matrix[j, :] - f), (self.item_matrix[j, :] - f))
            else:
                loss += (self.thau/2)*np.dot((self.item_matrix[j, :]), (self.item_matrix[j, :]))
            
        for i in range(self.num_user):
            loss += (self.thau/2)*np.dot(self.user_matrix[i, :], self.user_matrix[i, :])

        return loss, np.sqrt(rmse/count)

    @staticmethod
    def compute_latent(model, bound=(1, 20)):
        """
        Analysing The RMSE vesrus latent dimension:
        The dimension that result the minimum average loss will be selected
        Args: model: the model instance
              bound: the bind in which latent dim omputed
        Returns: the best latent dim and the corresponding RMSE
        """
        out = []
        for i in range(bound[0], bound[1]):
            model._reset_params()
            model.rmse_test = []
            model.rmse = []
            model.latent_dim = i
            model.fit(10)
            out.append([model.rmse_test])
        return out

    def _reset_params(self):
        #resetting the parameters of the model
        self.user_matrix = np.random.normal(0, 1/np.sqrt(self.latent_dim),size=(self.num_user, self.latent_dim))
        self.item_matrix = np.random.normal(0, 1/np.sqrt(self.latent_dim),size=(self.num_item, self.latent_dim))
        #Bias initialization
        self.user_bias = np.zeros((self.num_user))
        self.item_bias = np.zeros((self.num_item))
        
    def embbed_feature(self):
        """ Method to get the item sample given a feature """
        #read the data
        movie = pd.read_csv('data/movies.csv')
        #creating empty item to collect the sampled movie
        item_list = []
        #loop through all the movie and search all the movie item asscociated features
        for l in self.FEATURE.values():
            # genre = FEATURE[l]
            items = movie.loc[movie['genres'].str.contains(l)]
            items = items['movieId']
            idx = []
            #change the system idx to idx
            for item in items:
                if item in self.item_dict.keys():
                    idx.append(self.item_dict[item])
            #sample 5 items randomly
            items = pd.DataFrame(idx).sample(5)
            #store the sampled movie item
            item_list.append(items.to_numpy())

        return item_list

    def plot_feature(self):
        item_list = self.embbed_feature()
        for item in item_list:
            for i in item:
                plt.scatter(self.item_matrix[i[0], 0], self.item_matrix[i[0], 1])
        plt.legend(list(self.FEATURE.values()))
        
    def _get_features_dict(self):
        return  {   0:'Action',
                    1:'Adventure',
                    2:'Animation',
                    3:'Children',
                    4:'Comedy',
                    5:'Crime',
                    6:'Documentary',
                    7:'Drama',
                    8:'Fantasy',
                    9:'Film-Noir',
                    10:'Horror',
                    11:'Musical',
                    12:'Mystery',
                    13:'Romance',
                    14:'Sci-Fi',
                    15:'Thriller',
                    16:'War',
                    17:'Western',
                    18:'(no genres listed)',
                    19:'IMAX'
                }
