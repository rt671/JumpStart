import pickle

def saveModel(folder_path,file_name,matrixName,matrix):
        print("{}: Saving model in file '{}'".format(file_name, folder_path + file_name))
        dictionary_to_save = {matrixName: matrix}

        pickle.dump(dictionary_to_save,
                    open(folder_path + file_name, "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL)

        print("{}: Saving complete".format(file_name))