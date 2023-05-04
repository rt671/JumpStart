#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Cesare Bernardis
"""

import numpy as np
import scipy.sparse as sps
import pickle

from sklearn.preprocessing import normalize
from Base.Recommender_utils import check_matrix, similarityMatrixTopK

from Base.BaseSimilarityMatrixRecommender import BaseSimilarityMatrixRecommender
import time, sys

class RP3betaRecommender(BaseSimilarityMatrixRecommender):
    """ RP3beta recommender """

    RECOMMENDER_NAME = "Collaborative_Filtering"

    def __init__(self, URM_train):
        super(RP3betaRecommender, self).__init__(URM_train)


    def __str__(self):
        return "RP3beta(alpha={}, beta={}, min_rating={}, topk={}, implicit={}, normalize_similarity={})".format(self.alpha,
                                                                                        self.beta, self.min_rating, self.topK,
                                                                                        self.implicit, self.normalize_similarity)

    def fit(self, alpha=1., beta=0.6, min_rating=0, topK=100, implicit=False, normalize_similarity=True,firstTime=True,):

        self.alpha = alpha
        self.beta = beta
        self.min_rating = min_rating
        self.topK = topK
        self.implicit = implicit
        self.normalize_similarity = normalize_similarity
        self.firstTime=firstTime;

        
        # if X.dtype != np.float32:
        #     print("RP3beta fit: For memory usage reasons, we suggest to use np.float32 as dtype for the dataset")
        
        if self.min_rating > 0:
            self.URM_train.data[self.URM_train.data < self.min_rating] = 0
            self.URM_train.eliminate_zeros()
            if self.implicit:
                self.URM_train.data = np.ones(self.URM_train.data.size, dtype=np.float32)

        #Pui is the row-normalized urm
        Pui = normalize(self.URM_train, norm='l1', axis=1)
        # print("row-normalized urm:\n")
        # print(Pui.toarray());

        #Piu is the column-normalized, "boolean" urm transposed
        X_bool = self.URM_train.transpose(copy=True)
        X_bool.data = np.ones(X_bool.data.size, np.float32)

        # Taking the degree of each item to penalize top popular
        # Some rows might be zero, make sure their degree remains zero
        X_bool_sum = np.array(X_bool.sum(axis=1)).ravel()

        degree = np.zeros(self.URM_train.shape[1])

        nonZeroMask = X_bool_sum!=0.0

        degree[nonZeroMask] = np.power(X_bool_sum[nonZeroMask], -self.beta)
        # print("DEGREE:")
        # print(degree);

        #ATTENTION: axis is still 1 because i transposed before the normalization
        Piu = normalize(X_bool, norm='l1', axis=1)
        del(X_bool)

        # Alfa power
        if self.alpha != 1.:
            Pui = Pui.power(self.alpha)
            Piu = Piu.power(self.alpha)

        # Final matrix is computed as Pui * Piu * Pui
        # Multiplication unpacked for memory usage reasons
        block_dim = 200
        d_t = Piu
        # print("d_t:\n")
        # print(d_t.toarray(),"\n");
        W_sparse_mat=[];
        if(firstTime==True):
        # Use array as it reduces memory requirements compared to lists
            dataBlock = 10000000

            rows = np.zeros(dataBlock, dtype=np.int32)
            cols = np.zeros(dataBlock, dtype=np.int32)
            values = np.zeros(dataBlock, dtype=np.float32)

            numCells = 0

            start_time = time.time()
            start_time_printBatch = start_time
            # print(range(0, Pui.shape[1], block_dim));
            for current_block_start_row in range(0, Pui.shape[1], block_dim):

                if current_block_start_row + block_dim > Pui.shape[1]:
                    block_dim = Pui.shape[1] - current_block_start_row

                similarity_block = d_t[current_block_start_row:current_block_start_row + block_dim, :] * Pui
                similarity_block = similarity_block.toarray()
                # print("SIMILARITY_BLOCK:")
                # print(similarity_block,"\n");
                # print(block_dim);
                for row_in_block in range(block_dim):
                    row_data = np.multiply(similarity_block[row_in_block, :], degree)
                    row_data[current_block_start_row + row_in_block] = 0

                    best = row_data.argsort()[::-1][:self.topK]

                    notZerosMask = row_data[best] != 0.0

                    values_to_add = row_data[best][notZerosMask]
                    cols_to_add = best[notZerosMask]
                    # print(values_to_add);
                    for index in range(len(values_to_add)):

                        if numCells == len(rows):
                            rows = np.concatenate((rows, np.zeros(dataBlock, dtype=np.int32)))
                            cols = np.concatenate((cols, np.zeros(dataBlock, dtype=np.int32)))
                            values = np.concatenate((values, np.zeros(dataBlock, dtype=np.float32)))

                        rows[numCells] = current_block_start_row + row_in_block
                        cols[numCells] = cols_to_add[index]
                        values[numCells] = values_to_add[index]

                        numCells += 1
                    # print(rows);
                    # print(cols)
                    # print(values);
                    # print("\n");

                if time.time() - start_time_printBatch > 60:
                    print("Processed {} ( {:.2f}% ) in {:.2f} minutes. Rows per second: {:.0f}".format(
                        current_block_start_row,
                        100.0 * float(current_block_start_row) / Pui.shape[1],
                        (time.time() - start_time) / 60,
                        float(current_block_start_row) / (time.time() - start_time)))

                    sys.stdout.flush()
                    sys.stderr.flush()

                    start_time_printBatch = time.time()

            self.W_sparse = sps.csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])), shape=(Pui.shape[1], Pui.shape[1]))

        else:
            folder_path = "/Users/varunjain/Desktop/Jumpstart-BTP/matrices/"
            file_name = self.RECOMMENDER_NAME

            # load the saved dictionary
            saved_dict = pickle.load(open(folder_path + file_name, "rb"))

            # extract the W_sparse matrix from the saved dictionary
            self.W_sparse = saved_dict["W_sparse"]

            # check the type and shape of the loaded matrix
            print(type(self.W_sparse))    # should be <class 'scipy.sparse.csr.csr_matrix'>
            print(self.W_sparse.shape) 
            rows_changed=[5,7,0,3,89,6,600,800,8,9];
            for current_row in rows_changed:
                print(current_row);
                similarity_block = d_t[current_row:current_row + 1, :] * Pui
                similarity_block = similarity_block.toarray()
                print(similarity_block);
                # for row_in_block in range(block_dim):
                row_data = np.multiply(similarity_block[0, :], degree)
                print(row_data);
                row_data[current_row] = 0

                best = row_data.argsort()[::-1][:self.topK]

                notZerosMask = row_data[best] != 0.0

                values_to_add = row_data[best][notZerosMask]
                cols_to_add = best[notZerosMask]
                print(values_to_add);
                print(cols_to_add);
                for idx in range(len(cols_to_add)):
                    self.W_sparse[current_row,cols_to_add[idx]]=values_to_add[idx];

            # self.saveModel("/Users/varunjain/Desktop/Jumpstart-BTP/matrices/")

            # self.W_sparse = sps.csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])), shape=(Pui.shape[1], Pui.shape[1]))
            # print("RP3 beta-output:\n")
            # # print(self.W_sparse);
            # print(self.W_sparse.toarray());

        print("RP3 beta-output:\n")
            # print(self.W_sparse);
        print(self.W_sparse.toarray(),"\n");
        self.saveModel("/Users/varunjain/Desktop/Jumpstart-BTP/matrices/")

        if self.normalize_similarity:
            self.W_sparse = normalize(self.W_sparse, norm='l1', axis=1)


        if self.topK != False:
            self.W_sparse = similarityMatrixTopK(self.W_sparse, k=self.topK)
            
        self.W_sparse = check_matrix(self.W_sparse, format='csr')

    def saveModel(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        print("{}: Saving model in file '{}'".format(self.RECOMMENDER_NAME, folder_path + file_name))

        dictionary_to_save = {"W_sparse": self.W_sparse}


        pickle.dump(dictionary_to_save,
                    open(folder_path + file_name, "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL)

        print("{}: Saving complete".format(self.RECOMMENDER_NAME))
