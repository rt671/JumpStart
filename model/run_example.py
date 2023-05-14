from Collaborative_Filtering.RP3betaRecommender import RP3betaRecommender
from FeatureWeighting.Cython.Feature_Weighting import Feature_Weighting, EvaluatorCFW_D_wrapper
from Base.Evaluation.Evaluator import EvaluatorHoldout

# from Data_manager.Movielens_20m.Movielens20MReader import Movielens20MReader
from Data_manager.Movielens_1m.Movielens1MReader import Movielens1MReader
from Data_manager.DataSplitter_k_fold import DataSplitter_Warm_k_fold
from saveMatrices import saveModel
from gettopk import getTopK
import scipy.sparse as sps
import numpy as np
import pickle
from retrain_model import retrainModel

dataReader = Movielens1MReader()

    # Splitting the dataset. This split will produce a warm item split
    # To replicate the original experimens use the dataset accessible here with a cold item split:
    # https://mmprj.github.io/mtrm_dataset/index
dataSplitter = DataSplitter_Warm_k_fold(dataReader)
dataSplitter.load_data()

# Each URM is a scipy.sparse matrix of shape |users|x|items|
URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()
saveModel("/Users/varunjain/Desktop/Jumpstart-BTP/model/matrices/","URM_train","URM_train",URM_train)
saveModel("/Users/varunjain/Desktop/Jumpstart-BTP/model/matrices/","URM_validation","URM_validation",URM_validation)
saveModel("/Users/varunjain/Desktop/Jumpstart-BTP/model/matrices/","URM_test","URM_test",URM_test)

# The ICM is a scipy.sparse matrix of shape |items|x|features|
ICM = dataSplitter.get_ICM_from_name("ICM_genre")
print(ICM.shape);
saveModel("/Users/varunjain/Desktop/Jumpstart-BTP/model/matrices/","ICM","ICM",ICM)

# print("ICM:\n");
# print(ICM.toarray());
# print("URM:\n")
# print(URM_train.toarray());
retrainModel(2,[],[],True)

res=getTopK(4);
print(res)
# res=getTopK(3);
# print(res)
# res=getTopK(2);
# print(res)
