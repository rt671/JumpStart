import pickle
from Collaborative_Filtering.CF_Algorithm import CF_Algorithm
from FeatureWeighting.Cython.Feature_Weighting import Feature_Weighting, EvaluatorCFW_D_wrapper
from Base.Evaluation.Evaluator import EvaluatorHoldout
from saveMatrices import saveModel
from gettopk import getTopK
import pickle

def retrainModel(user_id_updated,item_id_array,rating_array,firstTime=False):

    print("Retrain running...")
    folder_path = "/Jumpstart/model/matrices/"
    file_name = "URM_train"
    saved_dict = pickle.load(open(folder_path + file_name, "rb"))
    URM_train = saved_dict["URM_train"]

    file_name = "URM_validation"
    saved_dict = pickle.load(open(folder_path + file_name, "rb"))
    URM_validation = saved_dict["URM_validation"]
   
    file_name = "URM_test"
    saved_dict = pickle.load(open(folder_path + file_name, "rb"))
    URM_test = saved_dict["URM_test"]
    
    file_name = "ICM"
    saved_dict = pickle.load(open(folder_path + file_name, "rb"))
    ICM = saved_dict["ICM"]

    k=0
    print(user_id_updated)
    print("ITEM-ID_ARRAY:")
    print(item_id_array)
    print(rating_array)
    print(URM_train.shape)
    liked_items=[]
    for items in item_id_array:
        item=items
        items=int(items)-1
        URM_train[(user_id_updated,items)]=rating_array[k]
        if(rating_array[k]==5):
            liked_items.append(item)
        k=k+1

    saveModel("/Jumpstart/model/matrices/","URM_train","URM_train",URM_train)

    # This contains the items to be ignored during the evaluation step
    # In a cold items setting this contains the indices of the warm items
    ignore_items = []

    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[5], ignore_items=ignore_items)
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[5], ignore_items=ignore_items)

    evaluator_validation_earlystopping = EvaluatorCFW_D_wrapper(evaluator_validation, ICM_target=ICM, model_to_use="last")
    firstTime=firstTime

    cf_parameters = {'topK': 500,
                    'alpha': 0.9,
                    'beta': 0.7,
                    'normalize_similarity': True,
                    'firstTime':firstTime,
                    'item_id_array':item_id_array}

    recommender_collaborative = CF_Algorithm(URM_train)
    recommender_collaborative.fit(**cf_parameters)

    # The similarity is a scipy.sparse matrix of shape |items|x|items|
    similarity_collaborative = recommender_collaborative.W_sparse.copy()

    # We instance and fit the feature weighting algorithm, it takes as input:
    # - The train URM
    # - The ICM
    # - The collaborative similarity matrix

    fw_parameters =  {'epochs': 5,
                    'learning_rate': 0.0001,
                    'sgd_mode': 'adam',
                    'add_zeros_quota': 1.0,
                    'l1_reg': 0.01,
                    'l2_reg': 0.001,
                    'topK': 100,
                    'use_dropout': True,
                    'dropout_perc': 0.7,
                    'initialization_mode_D': 'zero',
                    'positive_only_D': False,
                    'normalize_similarity': False,
                    'firstTime':firstTime,
                    'item_id_array':item_id_array}

    recommender_fw = Feature_Weighting(URM_train, ICM, similarity_collaborative)
    recommender_fw.fit(**fw_parameters,
                    evaluator_object=evaluator_validation_earlystopping,
                    stop_on_validation=True,
                    validation_every_n = 5,
                    lower_validations_allowed=10,
                    validation_metric="MAP")

    result_dict, result_string = evaluator_test.evaluateRecommender(recommender_fw)
    print("CFeCBF recommendation quality is: {}".format(result_string))
        
    ans=getTopK(user_id_updated,liked_items)
    return ans