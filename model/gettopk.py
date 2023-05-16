import pickle
import numpy as np
import random
from iim_collab import iim

def getTopK(user_id,item_id_array=[]):
    folder_path = "/Users/varunjain/Desktop/Jumpstart-BTP/model/matrices/"
    file_name = "Feature_Weighted_Content_Based"

    # load the saved dictionary
    saved_dict = pickle.load(open(folder_path + file_name, "rb"))

    # extract the W_sparse matrix from the saved dictionary
    W_sparse = saved_dict["W_sparse"]
    W_dense=W_sparse.toarray();
    print("IIM-content modified:")
    print(W_dense);
    file_name = "URM_train"

    # load the saved dictionary
    saved_dict = pickle.load(open(folder_path + file_name, "rb"))

    # extract the W_sparse matrix from the saved dictionary
    URM_train = saved_dict["URM_train"]
    
    print("URM_train in topK:::")
    print(URM_train.toarray());
    top_k = 15
    # num_users=5;

    # for user_id in range(num_users):
    user_history = URM_train[user_id, :]
    user_history_dense = user_history.toarray().flatten().astype(np.int32)
    user_row = W_dense[user_history_dense.nonzero()[0], :].mean(axis=0)

    print(user_row);

    top_k_idx = []
    print(user_history_dense.nonzero()[0]);
    for i in user_history_dense:
        idx = np.argsort(W_dense[:, i])[-top_k:]
        top_k_idx.append(idx)

    # top_k_idx.append(user_history_dense.nonzero()[0])
    # Combine indices and remove duplicates
    top_k_idx = np.unique(np.concatenate(top_k_idx))

    # Sort by decreasing order of similarity scores
    sorted_idx = np.argsort(W_dense[top_k_idx, :].dot(user_row))[::-1]

    # Return top K items
    top_k_items = top_k_idx[sorted_idx[:top_k]]+1
    print(top_k_items);
    iim(item_id_array,top_k_items);
    return top_k_items