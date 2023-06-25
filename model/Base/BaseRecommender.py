import numpy as np
import pickle, os
from Base.Recommender_utils import check_matrix

class BaseRecommender(object):
    RECOMMENDER_NAME = "Recommender_Base_Class"

    def __init__(self, URM_train):

        super(BaseRecommender, self).__init__()

        self.URM_train = check_matrix(URM_train.copy(), 'csr', dtype=np.float32)
        self.URM_train.eliminate_zeros()

        self.n_users, self.n_items = self.URM_train.shape

        self.normalize = False

        self.filterTopPop = False
        self.filterTopPop_ItemsID = np.array([], dtype=np.int)

        self.items_to_ignore_flag = False
        self.items_to_ignore_ID = np.array([], dtype=np.int)

        self._compute_item_score = self._compute_score_item_based
        self._URM_train_format_checked = False
        self._W_sparse_format_checked = False


    def fit(self):
        pass

    def get_URM_train(self):
        return self.URM_train.copy()

    def set_URM_train(self, URM_train_new, **kwargs):

        assert self.URM_train.shape == URM_train_new.shape, "{}: set_URM_train old and new URM train have different shapes".format(self.RECOMMENDER_NAME)

        if len(kwargs)>0:
            print("{}: set_URM_train keyword arguments not supported for this recommender class. Received: {}".format(self.RECOMMENDER_NAME, kwargs))

        self.URM_train = URM_train_new.copy()


    def set_items_to_ignore(self, items_to_ignore):
        self.items_to_ignore_flag = True
        self.items_to_ignore_ID = np.array(items_to_ignore, dtype=np.int)

    def saveModel(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        print("{}: Saving model in file '{}'".format(self.RECOMMENDER_NAME, folder_path + file_name))

        dictionary_to_save = {"W_sparse": self.W_sparse}


        pickle.dump(dictionary_to_save,
                    open(folder_path + file_name, "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL)

        print("{}: Saving complete".format(self.RECOMMENDER_NAME))
        

    def loadModel(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        print("{}: Loading model from file '{}'".format(self.RECOMMENDER_NAME, folder_path + file_name))


        data_dict = pickle.load(open(folder_path + file_name, "rb"))

        for attrib_name in data_dict.keys():
             self.__setattr__(attrib_name, data_dict[attrib_name])
        print("{}: Loading complete".format(self.RECOMMENDER_NAME))