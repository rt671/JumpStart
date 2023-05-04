import pickle

def saveModel(folder_path,file_name,matrixName,matrix):
        
        # if file_name is None:
        #     file_name = self.RECOMMENDER_NAME

        print("{}: Saving model in file '{}'".format(file_name, folder_path + file_name))

        # dictionary_to_save = {"W_sparse": self.W_sparse}
        dictionary_to_save = {matrixName: matrix}

        pickle.dump(dictionary_to_save,
                    open(folder_path + file_name, "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL)

        print("{}: Saving complete".format(file_name))