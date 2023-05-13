#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Maurizio Ferrari Dacrema
"""


import zipfile

from Data_manager.DataReader import DataReader
from Data_manager.DataReader_utils import downloadFromURL

import pandas as pd
import csv

class Movielens1MReader(DataReader):

    # DATASET_URL = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
    DATASET_SUBFOLDER = "Movielens_1m/"
    AVAILABLE_ICM = ["ICM_genre"]

    IS_IMPLICIT = True


    def __init__(self):
        super(Movielens1MReader, self).__init__()



    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER


    def _load_from_original_file(self):
        # Load data from original

        print("Movielens1MReader: Loading original data")

        zipFile_path = "/Users/varunjain/Desktop/ml-1m_changed.zip"

        try:

            # dataFile = zipfile.ZipFile(zipFile_path + "ml-1m.zip")
            dataFile = zipfile.ZipFile(zipFile_path)

        except (FileNotFoundError, zipfile.BadZipFile):

            print("Movielens1MReader: Unable to fild data zip file. Downloading...")

            # downloadFromURL(self.DATASET_URL, zipFile_path, "ml-1m.zip")

            dataFile = zipfile.ZipFile("/Users/varunjain/Desktop/ml-1m_changed.zip")
        
        print("READING DATASET...\n")
        genres_path = dataFile.extract("movies.csv", path=zipFile_path + "decompressed/")
        # # tags_path = dataFile.extract("ml-1m/tags.dat", path=zipFile_path + "decompressed/")
        URM_path = dataFile.extract("ratings.csv", path=zipFile_path + "decompressed/")
        
        self.tokenToFeatureMapper_ICM_genre = {}

        print("Movielens1MReader: loading genres")
        self.ICM_genre, self.tokenToFeatureMapper_ICM_genre, self.item_original_ID_to_index = self._loadICM_genres(genres_path, header=True, separator=',', genresSeparator="|")

        print("Movielens1MReader: loading URM")
        self.URM_all, _, self.user_original_ID_to_index = self._loadURM(URM_path, separator=",", header = True, if_new_user = "add", if_new_item = "ignore")


        print("Movielens1MReader: cleaning temporary files")

        import shutil

        shutil.rmtree(zipFile_path + "decompressed", ignore_errors=True)

        print("Movielens1MReader: saving URM and ICM")




    def _loadURM (self, filePath, header = False, separator="::", if_new_user = "add", if_new_item = "ignore"):

        from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix_FilterIDs

        URM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper = self.item_original_ID_to_index, on_new_col = if_new_item,
                                                        preinitialized_row_mapper = None, on_new_row = if_new_user)


        fileHandle = open(filePath, "r")
        numCells = 0

        if header:
            fileHandle.readline()

        for line in fileHandle:
            numCells += 1
            if (numCells % 1000000 == 0):
                print("Processed {} cells".format(numCells))

            if (len(line)) > 1:
                line = line.split(separator)

                line[-1] = line[-1].replace("\n", "")

            user_id = line[0]
            item_id = line[1]


            try:
                value = float(line[2])

                if value != 0.0:

                    URM_builder.add_data_lists([user_id], [item_id], [value])

            except:
                pass

        fileHandle.close()


        return  URM_builder.get_SparseMatrix(), URM_builder.get_column_token_to_id_mapper(), URM_builder.get_row_token_to_id_mapper()




    def _loadICM_genres(self, genres_path, header=True, separator=',', genresSeparator="|"):

        # Genres
        from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix_FilterIDs

        ICM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper = None, on_new_col = "add",
                                                        preinitialized_row_mapper = None, on_new_row = "add")
        fileHandle = open(genres_path, "r", encoding="latin1")
        numCells = 0

        if header:
            fileHandle.readline()

        for line in fileHandle:
            numCells += 1
            if (numCells % 1000000 == 0):
                print("Processed {} cells".format(numCells))

            if (len(line)) > 1:
                line = line.split(separator)

                line[-1] = line[-1].replace("\n", "")

                movie_id = line[0]

                # title = line[1]
                # In case the title contains commas, it is enclosed in "..."
                # genre list will always be the last element
                genreList = line[-1]

                genreList = genreList.split(genresSeparator)

                # Rows movie ID
                # Cols features
                ICM_builder.add_single_row(movie_id, genreList, data = 1.0)


        fileHandle.close()
        print("1st");
        print(ICM_builder.get_SparseMatrix());
        # print("2nd")
        # print(ICM_builder.get_column_token_to_id_mapper())
        # print("3rd")
        # print(ICM_builder.get_row_token_to_id_mapper())
        return ICM_builder.get_SparseMatrix(), ICM_builder.get_column_token_to_id_mapper(), ICM_builder.get_row_token_to_id_mapper()


# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on 14/09/17

# @author: Maurizio Ferrari Dacrema
# """



# import scipy.sparse as sps
# import zipfile

# from Data_manager.DataReader import DataReader
# from Data_manager.DataReader_utils import downloadFromURL, load_CSV_into_SparseBuilder




# class Movielens1MReader(DataReader):

#     DATASET_URL = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
#     DATASET_SUBFOLDER = "Movielens_1m/"
#     AVAILABLE_ICM = []
#     DATASET_SPECIFIC_MAPPER = []

#     IS_IMPLICIT = True


#     def __init__(self):
#         super(Movielens1MReader, self).__init__()


#     def _get_dataset_name_root(self):
#         return self.DATASET_SUBFOLDER



#     def _load_from_original_file(self):
#         # Load data from original

#         print("Movielens1MReader: Loading original data")

#         zipFile_path =  self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

#         try:

#             dataFile = zipfile.ZipFile(zipFile_path + "ml-1m.zip")

#         except (FileNotFoundError, zipfile.BadZipFile):

#             print("Movielens1MReader: Unable to fild data zip file. Downloading...")

#             downloadFromURL(self.DATASET_URL, zipFile_path, "ml-1m.zip")

#             dataFile = zipfile.ZipFile(zipFile_path + "ml-1m.zip")


#         URM_path = dataFile.extract("ml-1m/ratings.dat", path=zipFile_path + "decompressed/")

#         self.URM_all, self.item_original_ID_to_index, self.user_original_ID_to_index = load_CSV_into_SparseBuilder(URM_path, separator="::")


#         print("Movielens1MReader: cleaning temporary files")

#         import shutil

#         shutil.rmtree(zipFile_path + "decompressed", ignore_errors=True)

#         print("Movielens1MReader: loading complete")
