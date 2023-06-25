import pickle
import numpy as np
import random

def iim(item_id_array,top_k_items):

    folder_path = "/Jumpstart/model/matrices/"
    file_name = "Collaborative_Filtering"

    # load the saved dictionary
    saved_dict = pickle.load(open(folder_path + file_name, "rb"))

    # extract the W_sparse matrix from the saved dictionary
    W_sparse = saved_dict["W_sparse"]
    W_dense=W_sparse.toarray()
    print("Item-id-array in iim-collab:")
    print(item_id_array)
    print("IIM-Collab:\n")
    for row in item_id_array:
        row = int(row)
        print(row)
        print(W_dense[row])
        sorted_indices = sorted(range(len(W_dense[row])), key=lambda idx: W_dense[row][idx], reverse=True)
        length = 0
        for idx in sorted_indices:
            if W_dense[row][idx] != 0:
                length += 1
                if length >=5 and length<=10:
                    random_value = random.choice(top_k_items)
                    print(random_value, W_dense[row][idx])
                else:
                    print(idx, W_dense[row][idx])
                if length > 20:
                    break
        print("\n")