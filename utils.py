import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd

class Utils:
    """Util calss contains extra helper functions for the main model class"""

    def plot(self, file_name, plot, save=True):
        #Lets only plot RMSE..[Useful metrics]
        fig, ax = plt.subplots(figsize=(14,6))
        # plt.subplot(1, 2, 1)  # row 1, column 2, count 1
        # plt.plot(np.arange(0, len(self.rmse),1), np.array(self.rmse), color='g')
        # plt.plot(np.arange(0, len(self.rmse_test),1), np.array(self.rmse_test), color='r')
        # plt.legend(['training rmse', 'test rmse'])
        # plt.xlabel('# iterations')
        # plt.ylabel('RMSE')
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
        with open('item_feature_dict.pkl', 'rb') as f:
            dicts = pickle.load(f)
        return dicts


    def get_items(self, contains_feature=True):
        """ 
        Number of movies for a given genre 
        Args: l: feature(genre index)
              contains_feature: Used to return the movies that corresponding to a 
              certain feature if it is True else it returns that the movie that not 
              corresponding to a given feature
        """
        # Extrat item given genres contains in csv file
        if contains_feature:
            with open('feature_item_dict_True.pkl', 'rb') as f:
                dicts = pickle.load(f)
        else:
            with open('feature_item_dict_False.pkl', 'rb') as f:
                dicts = pickle.load(f)
        # System Id for a given generes
        return dicts

    def compute_cost(self, data):
        loss = 0
        rmse = 0
        count = 0
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
        
    def compute_latent(self, bound=(1, 20)):
        """
        Analysing The RMSE vesrus latent dimension:
        The dimension that result the minimum average loss will be selected
        Args: model: the model instance
              bound: the bind in which latent dim omputed
        Returns: the best latent dim and the corresponding RMSE
        """
        out = []
        for i in range(bound[0], bound[1]):
            self.latent_dim = i
            self.fit(20)
            out.append((i, min(mean(self.rmse_test))))
        return out