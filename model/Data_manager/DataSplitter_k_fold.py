import scipy.sparse as sps
import numpy as np
import pickle
import traceback, os

class DataSplitter_k_fold(object):
    DATASET_SPLIT_ROOT_FOLDER = "Data_manager_split_datasets/"
    ICM_SPLIT_SUFFIX = [""]

    def __init__(self, dataReader_object, n_folds = 5, forbid_new_split = False, force_new_split = False):
        self.n_folds = n_folds
        self.dataReader_object = dataReader_object
        self.forbid_new_split = forbid_new_split
        self.force_new_split = force_new_split

        super(DataSplitter_k_fold, self).__init__()


    def get_statistics_URM(self):
        n_items = int(self.n_items)
        n_users = int(self.n_users)

        print("DataSplitter_k_fold for DataReader: {}\n"
              "\t Num items: {}\n"
              "\t Num users: {}\n".format(self.dataReader_object._get_dataset_name(), n_items, n_users))

        n_global_interactions = 0

        for fold_index in range(self.n_folds):
            URM_fold_object = self.fold_split[fold_index]["URM"]
            n_global_interactions += URM_fold_object.nnz


        for fold_index in range(self.n_folds):
            URM_fold_object = self.fold_split[fold_index]["URM"]
            items_in_fold = self.fold_split[fold_index]["items_in_fold"]


            # print("\t Statistics for fold {}: n_interactions {} ( {:.2f}%), n_items {} ( {:.2f}%), density: {:.2E}".format(
            #     fold_index,
            #     URM_fold_object.nnz, URM_fold_object.nnz/n_global_interactions*100,
            #     len(items_in_fold), len(items_in_fold)/n_items*100,
            #     URM_fold_object.nnz/(int(n_items)*int(n_users))
            # ))

        print("\n")

    def get_dataReader_object(self):
        return self.dataReader_object

    # Allow to use ICM functions on the DataSplitter
    def _get_dataset_name(self):
        return self.get_dataReader_object()._get_dataset_name()

    def get_ICM_from_name(self, ICM_name):
        return getattr(self, ICM_name).copy()

    def get_loaded_ICM_names(self):
        return self.get_dataReader_object().get_loaded_ICM_names()

    def get_all_available_ICM_names(self):
        return self.get_dataReader_object().get_all_available_ICM_names().copy()


    def get_loaded_ICM_dict(self):
        return self.get_dataReader_object().get_loaded_ICM_dict()

    def load_data(self, save_folder_path = None):

        # Use default "dataset_name/split_name/original" or "dataset_name/split_name/k-cores"
        if save_folder_path is None:
            save_folder_path = self.DATASET_SPLIT_ROOT_FOLDER + \
                               self.dataReader_object._get_dataset_name_root() + \
                               self._get_split_subfolder_name() + \
                               self.dataReader_object._get_dataset_name_data_subfolder()


        # If save_folder_path contains any path try to load a previously built split from it
        if save_folder_path is not False and not self.force_new_split:

            try:

                self._load_previously_built_split_and_attributes(save_folder_path)

            except FileNotFoundError:

                # Split not found, either stop or create a new one
                if self.forbid_new_split:
                    raise ValueError("DataSplitter_k_fold: Preloaded data not found, but creating a new split is forbidden. Terminating")

                else:
                    print("DataSplitter_k_fold: Preloaded data not found, reading from original files...")

                    # If directory does not exist, create
                    if not os.path.exists(save_folder_path):
                        os.makedirs(save_folder_path)

                    self._split_data_from_original_dataset(save_folder_path)
                    self._load_previously_built_split_and_attributes(save_folder_path)

                    print("DataSplitter_k_fold: Preloaded data not found, reading from original files... Done")


            except Exception:

                print("DataSplitter_k_fold: Reading split from {} caused the following exception...".format(save_folder_path))
                traceback.print_exc()
                raise Exception("DataSplitter_k_fold: Exception while reading split")
            
        self.get_statistics_URM()
        self.get_statistics_ICM()

        print("DataSplitter_k_fold: Done.")

    def get_statistics_ICM(self):

        for ICM_name in self.dataReader_object.get_loaded_ICM_names():

            ICM_object = getattr(self, ICM_name)
            n_items = ICM_object.shape[0]
            n_features = ICM_object.shape[1]

            print("\t Statistics for {}: n_features {}, feature occurrences {}, density: {:.2E}".format(
                ICM_name, n_features, ICM_object.nnz, ICM_object.nnz/(int(n_items)*int(n_features))
            ))

        print("\n")

    def get_fold_split(self):
        return self.fold_split


    def get_fold(self, n_fold):
        return self.fold_split[n_fold]["URM"].copy()



    def get_URM_train_for_test_fold(self, n_test_fold):
        """
        The train set is defined as all data except the one of that fold, which is the test
        :param n_fold:
        :return:
        """

        URM_test = self.fold_split[n_test_fold]["URM"].copy()

        URM_train = sps.csr_matrix(URM_test.shape)

        for fold_index in range(self.n_folds):

            if fold_index != n_test_fold:
                URM_fold_object = self.fold_split[fold_index]["URM"]

                URM_train+=URM_fold_object


        return URM_train, URM_test



    def get_holdout_split(self):
        """
        The train set is defined as all data except the one of that fold, which is the test
        :param n_fold:
        :return:
        """

        assert self.n_folds >= 3, "DataSplitter: To get a holdout split URM_train, URM_validation, URM_test, the splitter must have at least 3 folds, currently it has {}".format(self.n_folds)

        print("DataSplitter: Generating holdout split... ")

        URM_test = self.fold_split[0]["URM"].copy()
        URM_validation = self.fold_split[1]["URM"].copy()


        URM_train = sps.csr_matrix(URM_test.shape)

        for fold_index in range(2, self.n_folds):

            URM_fold_object = self.fold_split[fold_index]["URM"]

            URM_train+=URM_fold_object


        print("DataSplitter: Generating holdout split... done!")

        return URM_train, URM_validation, URM_test


    def __iter__(self):

        self.__iterator_current_fold = 0
        return self


    def __next__(self):

        fold_to_return = self.__iterator_current_fold

        if self.__iterator_current_fold >= self.n_folds:
            raise StopIteration

        self.__iterator_current_fold += 1

        return fold_to_return, self[fold_to_return]


    def __getitem__(self, n_test_fold):
        """
        :param index:
        :return:
        """

        return self.get_URM_train_for_test_fold(n_test_fold)


    def __len__(self):

        return self.n_folds


class DataSplitter_Warm_k_fold(DataSplitter_k_fold):
    """
    This splitter performs a Holdout from the full URM splitting in train, test and validation
    Ensures that every user has at least an interaction in all splits
    """

    def __init__(self, dataReader_object, n_folds = 5, forbid_new_split = False,
                 allow_cold_users = False):

        self.allow_cold_users = allow_cold_users

        super(DataSplitter_Warm_k_fold, self).__init__(dataReader_object,
                                                       n_folds=n_folds,
                                                       forbid_new_split = forbid_new_split)



    def _get_split_subfolder_name(self):
        return "warm_{}_fold/".format(self.n_folds)



    def _split_data_from_original_dataset(self, save_folder_path):
        self.dataReader_object.load_data()
        URM = self.dataReader_object.get_URM_all()
        URM = sps.csr_matrix(URM)



        # if not self.allow_cold_users:
        #     user_interactions = np.ediff1d(URM.indptr)
        #     user_to_preserve = user_interactions >= self.n_folds

        #     print("DataSplitter_Warm: Removing {} of {} users because they have less interactions than the number of folds".format(
        #          URM.shape[0] - user_to_preserve.sum(), URM.shape[0]))

        #     URM = URM[user_to_preserve,:]


        self.n_users, self.n_items = URM.shape


        URM = sps.csr_matrix(URM)

        # Create empty URM for each fold
        self.fold_split = {}

        for fold_index in range(self.n_folds):
            self.fold_split[fold_index] = {}
            self.fold_split[fold_index]["URM"] = sps.coo_matrix(URM.shape)

            URM_fold_object = self.fold_split[fold_index]["URM"]
            # List.extend is waaaay faster than numpy.concatenate
            URM_fold_object.row = []
            URM_fold_object.col = []
            URM_fold_object.data = []


        for user_id in range(self.n_users):

            start_user_position = URM.indptr[user_id]
            end_user_position = URM.indptr[user_id+1]

            user_profile = URM.indices[start_user_position:end_user_position]

            indices_to_suffle = np.arange(len(user_profile), dtype=np.int)

            np.random.shuffle(indices_to_suffle)

            user_profile = user_profile[indices_to_suffle]
            user_interactions = URM.data[start_user_position:end_user_position][indices_to_suffle]



            # interactions_per_fold is a float number, to auto-adjust fold size
            interactions_per_fold = len(user_profile)/self.n_folds

            for fold_index in range(self.n_folds):

                start_pos = int(interactions_per_fold*fold_index)
                end_pos = int(interactions_per_fold*(fold_index+1))

                if fold_index == self.n_folds-1:
                    end_pos = len(user_profile)

                current_fold_user_profile = user_profile[start_pos:end_pos]
                current_fold_user_interactions = user_interactions[start_pos:end_pos]

                URM_fold_object = self.fold_split[fold_index]["URM"]

                URM_fold_object.row.extend([user_id]*len(current_fold_user_profile))
                URM_fold_object.col.extend(current_fold_user_profile)
                URM_fold_object.data.extend(current_fold_user_interactions)


        for fold_index in range(self.n_folds):
            URM_fold_object = self.fold_split[fold_index]["URM"]
            URM_fold_object.row = np.array(URM_fold_object.row, dtype=np.int)
            URM_fold_object.col = np.array(URM_fold_object.col, dtype=np.int)
            URM_fold_object.data = np.array(URM_fold_object.data, dtype=np.float)

            self.fold_split[fold_index]["URM"] = sps.csr_matrix(URM_fold_object)
            self.fold_split[fold_index]["items_in_fold"] = np.arange(0, self.n_items, dtype=np.int)


        fold_dict_to_save = {"fold_split": self.fold_split,
                             "n_folds": self.n_folds,
                             "n_items": self.n_items,
                             "n_users": self.n_users,
                             "allow_cold_users": self.allow_cold_users,
                             }

        if self.allow_cold_users:
            allow_user = "allow_cold_users"
        else:
            allow_user = "only_warm_users"

        pickle.dump(fold_dict_to_save,
                    open(save_folder_path + "URM_{}_fold_split_{}".format(self.n_folds, allow_user), "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL)





        for ICM_name in self.dataReader_object.get_loaded_ICM_names():

            pickle.dump(self.dataReader_object.get_ICM_from_name(ICM_name),
                        open(save_folder_path + "{}".format(ICM_name), "wb"),
                        protocol=pickle.HIGHEST_PROTOCOL)

            pickle.dump(self.dataReader_object.get_ICM_feature_to_index_mapper_from_name(ICM_name),
                        open(save_folder_path + "tokenToFeatureMapper_{}".format(ICM_name), "wb"),
                        protocol=pickle.HIGHEST_PROTOCOL)

        print("DataSplitter: Split complete")

    def _load_previously_built_split_and_attributes(self, save_folder_path):
        # Loads all URM and ICM

        if self.allow_cold_users:
            allow_cold_users_file_name = "allow_cold_users"
        else:
            allow_cold_users_file_name = "only_warm_users"


        data_dict = pickle.load(open(save_folder_path + "URM_{}_fold_split_{}".format(self.n_folds, allow_cold_users_file_name), "rb"))

        for attrib_name in data_dict.keys():
             self.__setattr__(attrib_name, data_dict[attrib_name])


        for ICM_name in self.dataReader_object.get_loaded_ICM_names():

            ICM_object = pickle.load(open(save_folder_path + "{}".format(ICM_name), "rb"))
            self.__setattr__(ICM_name, ICM_object)