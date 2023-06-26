from Data_manager.Movielens1MReader import Movielens1MReader
from Data_manager.DataSplitter_k_fold import DataSplitter_Warm_k_fold
from saveMatrices import saveModel
from gettopk import getTopK
from retrain_model import retrainModel

dataReader = Movielens1MReader()
dataSplitter = DataSplitter_Warm_k_fold(dataReader)
dataSplitter.load_data()

URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()
saveModel("/Jumpstart/model/matrices/","URM_train","URM_train",URM_train)
saveModel("/Jumpstart/model/matrices/","URM_validation","URM_validation",URM_validation)
saveModel("/Jumpstart/model/matrices/","URM_test","URM_test",URM_test)

# The ICM is a scipy.sparse matrix of shape |items|x|features|
ICM = dataSplitter.get_ICM_from_name("ICM_genre")
print(URM_train.shape)
saveModel("/Jumpstart/model/matrices/","ICM","ICM",ICM)

retrainModel(2,[],[],True)

res=getTopK(4)
print(res)