from Base.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Base.Recommender_utils import check_matrix
from Base.IR_feature_weighting import okapi_BM_25, TF_IDF
from Base.Similarity.Compute_Similarity import Compute_Similarity
from Base.BaseSimilarityMatrixRecommender import BaseSimilarityMatrixRecommender
from CythonCompiler.run_compile_subprocess import run_compile_subprocess

import time, sys
import numpy as np

class EvaluatorCFW_D_wrapper(object):

    def __init__(self, evaluator_object, ICM_target, model_to_use = "best"):

        self.evaluator_object = evaluator_object
        self.ICM_target = ICM_target.copy()

        assert model_to_use in ["best", "last"], "EvaluatorCFW_D_wrapper: model_to_use must be either 'best' or 'incremental'. Provided value is: '{}'".format(model_to_use)

        self.model_to_use = model_to_use
        

    def evaluateRecommender(self, recommender_object):

        recommender_object.set_ICM_and_recompute_W(self.ICM_target, recompute_w = False)

        # Use either best model or incremental one
        recommender_object.compute_W_sparse(model_to_use = self.model_to_use)
        return self.evaluator_object.evaluateRecommender(recommender_object)


class Feature_Weighting(BaseSimilarityMatrixRecommender,Incremental_Training_Early_Stopping):

    RECOMMENDER_NAME = "Feature_Weighted_Content_Based"

    INIT_TYPE_VALUES = ["random", "one", "BM25", "TF-IDF"]

    def __init__(self, URM_train, ICM, sim_matrix_target):

        super(Feature_Weighting, self).__init__(URM_train)

        # if (URM_train.shape[1] != ICM.shape[0]):
        #     raise ValueError("Number of items not consistent. URM contains {} but ICM contains {}".format(URM_train.shape[1], ICM.shape[0]))

        # if(sim_matrix_target.shape[0] != sim_matrix_target.shape[1]):
        #     raise ValueError("Items imilarity matrix is not square: rows are {}, columns are {}".format(sim_matrix_target.shape[0], sim_matrix_target.shape[1]))

        # if(sim_matrix_target.shape[0] != ICM.shape[0]):
        #     raise ValueError("Number of items not consistent. S_matrix contains {} but ICM contains {}".format(sim_matrix_target.shape[0], ICM.shape[0]))

        self.sim_matrix_target = check_matrix(sim_matrix_target, 'csr')
        print("SIM_MAT_TARGET:")
        print(sim_matrix_target.toarray());
        self.ICM = check_matrix(ICM, 'csr')
        self.n_features = self.ICM.shape[1]


    def fit(self, precompute_common_features = False,
            learning_rate = 0.1,
            positive_only_D = True,
            initialization_mode_D ="random",
            normalize_similarity = False,
            use_dropout = True,
            dropout_perc = 0.3,
            l1_reg = 0.0,
            l2_reg = 0.0,
            epochs = 50,
            topK = 300,
            add_zeros_quota = -1.0,
            verbose = False,
            sgd_mode = 'adagrad', gamma = 0.9, beta_1 = 0.9, beta_2 = 0.999,
            firstTime=False,
            **earlystopping_kwargs):

        if initialization_mode_D not in self.INIT_TYPE_VALUES:
           raise ValueError("Value for 'initialization_mode_D' not recognized. Acceptable values are {}, provided was '{}'".format(self.INIT_TYPE_VALUES, initialization_mode_D))

        self.normalize_similarity = normalize_similarity
        self.learning_rate = learning_rate
        self.add_zeros_quota = add_zeros_quota
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.epochs = epochs
        self.topK = topK
        self.verbose = verbose
        self.firstTime=firstTime

        weights_initialization_D = None

        if initialization_mode_D == "random":
            weights_initialization_D = np.random.normal(0.001, 0.1, self.n_features).astype(np.float64)
        elif initialization_mode_D == "one":
            weights_initialization_D = np.ones(self.n_features, dtype=np.float64)
        elif initialization_mode_D == "zero":
            weights_initialization_D = np.zeros(self.n_features, dtype=np.float64)
        elif initialization_mode_D == "BM25":
            weights_initialization_D = np.ones(self.n_features, dtype=np.float64)
            self.ICM = self.ICM.astype(np.float32)
            self.ICM = okapi_BM_25(self.ICM)
        elif initialization_mode_D == "TF-IDF":
            weights_initialization_D = np.ones(self.n_features, dtype=np.float64)
            self.ICM = self.ICM.astype(np.float32)
            self.ICM = TF_IDF(self.ICM)
        else:
            raise ValueError("CFW_D_Similarity_Cython: 'init_type' not recognized")

        self._generate_train_data()

        # Import compiled module
        from FeatureWeighting.Cython.FW_Algorithm import FW_Algorithm
        
        # Instantiate fast Cython implementation
        self.FW_D_Similarity = FW_Algorithm(self.row_list, self.col_list, self.data_list,
                                                      self.n_features, self.ICM,
                                                      precompute_common_features = precompute_common_features,
                                                      positive_only_D = positive_only_D,
                                                      weights_initialization_D = weights_initialization_D,
                                                      use_dropout = use_dropout, dropout_perc = dropout_perc,
                                                      learning_rate=learning_rate, 
                                                      l1_reg=l1_reg, l2_reg=l2_reg,
                                                      sgd_mode=sgd_mode, 
                                                      verbose=self.verbose,
                                                      gamma = gamma, beta_1=beta_1, beta_2=beta_2)

        if self.verbose:
            print(self.RECOMMENDER_NAME + ": Initialization completed")

        self.D_incremental = self.FW_D_Similarity.get_weights()
        self.D_best = self.D_incremental.copy()

        self._train_with_early_stopping(epochs,
                                        algorithm_name = self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)

        self.compute_W_sparse(model_to_use = "best")

        sys.stdout.flush()


    def _prepare_model_for_validation(self):
        self.D_incremental = self.FW_D_Similarity.get_weights()
        self.compute_W_sparse(model_to_use = "last")

    def _update_best_model(self):
        self.D_best = self.D_incremental.copy()

    def _run_epoch(self, num_epoch):
        self.loss = self.FW_D_Similarity.fit()

    def _generate_train_data(self):
        if self.verbose:
            print(self.RECOMMENDER_NAME + ": Generating train data")

        start_time_batch = time.time()

        self.similarity = Compute_Similarity(self.ICM.T, shrink=0, topK=self.topK, normalize=False)
        sim_mat_content = self.similarity.compute_similarity()
        sim_mat_content = check_matrix(sim_mat_content, "csr")
        print("SIM_MAT_CONTENT:")
        print(sim_mat_content.toarray());

        self.write_log("Collaborative S density: {:.2E}, nonzero cells {}".format(
            self.sim_matrix_target.nnz/self.sim_matrix_target.shape[0]**2, self.sim_matrix_target.nnz))

        self.write_log("Content S density: {:.2E}, nonzero cells {}".format(
            sim_mat_content.nnz/sim_mat_content.shape[0]**2, sim_mat_content.nnz))

        if self.normalize_similarity:
            sum_of_squared_features = np.array(self.ICM.T.power(2).sum(axis=0)).ravel()
            sum_of_squared_features = np.sqrt(sum_of_squared_features)

        num_common_coordinates = 0

        estimated_n_samples = int(sim_mat_content.nnz*(1+self.add_zeros_quota)*1.2)

        self.row_list = np.zeros(estimated_n_samples, dtype=np.int32)
        self.col_list = np.zeros(estimated_n_samples, dtype=np.int32)
        self.data_list = np.zeros(estimated_n_samples, dtype=np.float64)

        num_samples = 0

        row_change_list=[3,5]
        # col_change_list=[]
        if self.firstTime == True :
            for row_index in range(self.n_items):
                num_samples=self.find_similar_items(row_index,sim_mat_content,num_common_coordinates,estimated_n_samples,num_samples)

        else:
            # array  of rows and array of columns
            for row_index in row_change_list:
                num_samples=self.find_similar_items(row_index,sim_mat_content,num_common_coordinates,estimated_n_samples,num_samples)
            # for col_index in col_change_list:
            #     num_samples=self.find_similar_items(col_index,sim_mat_content,num_common_coordinates,estimated_n_samples,num_samples)

        if self.verbose and (time.time() - start_time_batch > 30 or num_samples == sim_mat_content.nnz*(1+self.add_zeros_quota)):

            print(self.RECOMMENDER_NAME + ": Generating train data. Sample {} ( {:.2f} %) ".format(
                    num_samples, num_samples/ sim_mat_content.nnz*(1+self.add_zeros_quota) *100))

            sys.stdout.flush()
            sys.stderr.flush()

            start_time_batch = time.time()


        self.write_log("Content S structure has {} out of {} ( {:.2f}%) nonzero collaborative cells".format(
            num_common_coordinates, sim_mat_content.nnz, num_common_coordinates/sim_mat_content.nnz*100))

        # Discarding extra cells
        self.row_list = self.row_list[:num_samples]
        self.col_list = self.col_list[:num_samples]
        self.data_list = self.data_list[:num_samples]
        # print("ALL LIST:")
        # print(self.row_list)
        # print(self.col_list)
        # print(self.data_list)

        data_nnz = sum(np.array(self.data_list)!=0) # number of non-zero cells in data_list
        data_sum = sum(self.data_list)

        collaborative_nnz = self.sim_matrix_target.nnz
        collaborative_sum = sum(self.sim_matrix_target.data)

        # self.write_log("Nonzero collaborative cell sum is: {:.2E}, average is: {:.2E}, "
        #               "average over all collaborative data is {:.2E}".format(
        #               data_sum, data_sum/data_nnz, collaborative_sum/collaborative_nnz))

    def find_similar_items(self, row_index,sim_mat_content,num_common_coordinates,estimated_n_samples,num_samples):

            start_pos_content = sim_mat_content.indptr[row_index]
            end_pos_content = sim_mat_content.indptr[row_index+1]

            content_coordinates = sim_mat_content.indices[start_pos_content:end_pos_content]

            start_pos_target = self.sim_matrix_target.indptr[row_index]
            end_pos_target = self.sim_matrix_target.indptr[row_index+1]

            target_coordinates = self.sim_matrix_target.indices[start_pos_target:end_pos_target]

            is_common = np.in1d(content_coordinates, target_coordinates)
            # print(is_common);
            num_common_in_current_row = is_common.sum()
            num_common_coordinates += num_common_in_current_row

            for index in range(len(is_common)):
                random_value = np.random.rand()
                # print("RANDOM_VALUE:\n")
                # print(random_value)
                if num_samples == estimated_n_samples:
                    dataBlock = 1000000
                    self.row_list = np.concatenate((self.row_list, np.zeros(dataBlock, dtype=np.int32)))
                    self.col_list = np.concatenate((self.col_list, np.zeros(dataBlock, dtype=np.int32)))
                    self.data_list = np.concatenate((self.data_list, np.zeros(dataBlock, dtype=np.float64)))

                if is_common[index]:
                    # If cell exists in target matrix, add its value
                    # Otherwise it will remain zero with a certain probability

                    col_index = content_coordinates[index]

                    self.row_list[num_samples] = row_index
                    self.col_list[num_samples] = col_index
                    # print("Row and Column is: ", row_index, " ", col_index)
                    # print("\n")
                    
                    # print(self.sim_matrix_target.toarray(),"\n");
                    new_data_value = self.sim_matrix_target[row_index, col_index]
                    # print("NEW_DATA:\n")
                    # print(new_data_value)

                    # if self.normalize_similarity:
                    #     new_data_value *= sum_of_squared_features[row_index]*sum_of_squared_features[col_index]

                    self.data_list[num_samples] = new_data_value
                    num_samples += 1

                # elif np.random.rand()<= self.add_zeros_quota:

                #     col_index = content_coordinates[index]

                #     self.row_list[num_samples] = row_index
                #     self.col_list[num_samples] = col_index
                #     self.data_list[num_samples] = 0.0

                    num_samples += 1
            return num_samples        

    def write_log(self, string):
        if self.verbose:
            print(self.RECOMMENDER_NAME + ": " + string)
            sys.stdout.flush()
            sys.stderr.flush()


    def compute_W_sparse(self, model_to_use = "best"):

        if model_to_use == "last":
            feature_weights = self.D_incremental
        elif model_to_use == "best":
            feature_weights = self.D_best
        else:
            assert False, "{}: compute_W_sparse, 'model_to_use' parameter not recognized".format(self.RECOMMENDER_NAME)

        self.similarity = Compute_Similarity(self.ICM.T, shrink=0, topK=self.topK,
                                            normalize=self.normalize_similarity, row_weights=feature_weights)

        self.W_sparse = self.similarity.compute_similarity()
        self.W_sparse = check_matrix(self.W_sparse, format='csr')


    def set_ICM_and_recompute_W(self, ICM_new, recompute_w = True):

        self.ICM = ICM_new.copy()

        if recompute_w:
            self.compute_W_sparse(model_to_use = "best")