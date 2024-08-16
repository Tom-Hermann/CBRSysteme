import pandas as pd
import numpy as np

from sys import stderr


# create a case base reasoning system

class myCBRSystem:

    # INITIALIZATION
    def __init__(self, case_base: pd.DataFrame):
        self.case_base = case_base

        # add case id: format "ID-<index>"

        self.case_base.insert(0, 'case_id', [f"ID-{i}" for i in range(len(self.case_base))])

        self.similarity_funct = {column: self.get_function_from_type(type) for column, type in case_base.dtypes.items() if column != 'case_id'}
        self.similarity_matrix = {}

        self.additionnal_functions_variable = {}
        self.weights = {column: 1 for column in case_base.columns if column != 'case_id'}

    def get_function_from_type(self, type):
        match type:
            case 'float64' | 'int64':
                return self.number_similarity
            case 'datetime64[ns]':
                return self.date_similarity
            case 'str':
                return self.matrix_similarity
            case 'object':
                return self.matrix_similarity
            case _:
                print("Type not recognized:", type)
                return None

    # SIMILARITY FUNCTION

    def number_similarity(self, x, y, name_column):

        # get similarity between two number normalized between 0 and 1

        # get the maximum value
        max_value = self.case_base[name_column].max()
        # get the minimum value
        min_value = self.case_base[name_column].min()

        # get the similarity
        similarity = 1 - abs(x - y) / (max_value - min_value)

        return similarity

    def date_similarity(self, x: pd.Timestamp, y: pd.Timestamp, name_column: str):
        # get similarity between to date normalized between 0 and 1

        # get the difference in days
        delta = (x - y).days
        # get the maximum difference in days
        max_delta = (self.case_base[name_column].max() - self.case_base[name_column].min()).days
        # get the similarity
        similarity = 1 - delta / max_delta


        return similarity

    def matrix_similarity(self, x, y, name_column):
        if not (name_column in self.similarity_matrix) and self.weights[name_column] != 0:
            print("matrix set: ", self.similarity_matrix)
            raise NotImplementedError(f"Matrix distance not set for value {name_column}")
        else:
            try:
                matrix = self.similarity_matrix[name_column]
                return_value = matrix[x][y]
            except IndexError:
                print("Value not in matrix, failed matrix distance.\nThe matrix is:", matrix, file=stderr)
                raise ValueError("Value not in matrix")
            return return_value

    # GET FUNCTION



    def get_similarity(self, case1: pd.Series, case2: pd.Series):
        similarity = 0
        for value1, value2, column in zip(case1, case2[1:], case1.index):
            if self.weights[column] == 0 or column == 'case_id':
                continue
            # similarity += self.weights[column] * (self.similarity_funct[column])(value1, value2, column)
            try:
                similarity += self.weights[column] * (self.similarity_funct[column])(value1, value2, column)
            except Exception as e:
                print(f"Error similarity mesure for {column}. Similarity function : {self.similarity_funct[column].__name__}. Weight: {self.weights[column]}")
                similarity += 0
                raise e
        try:
            similarity = similarity / sum(self.weights.values())
        except ZeroDivisionError:
            print("All weights are set to 0, no similarity can be computed", file=stderr)
            return 0

        return similarity


    def get_similaritys(self, case: pd.Series):

        if type(case) != pd.Series:
            print("Case must be a single case", file=stderr)
            return None
        self.add_case(case)

        # use the self.get_similarity for every record in self.case_base, return a dataframe with (id, result fo get_similarity)
        similarity_df = self.case_base.apply(lambda x: (x.case_id, self.get_similarity(case, x)), axis=1)
        similarity_df = pd.DataFrame(similarity_df.to_list(), columns=['case_id', 'similarity'])
        return similarity_df


    def get_columns(self):
        return self.case_base.columns

    def get_functions(self):
        return {column: function.__name__ if function != None else "None" for column, function in self.similarity_funct.items()}

    def get_cases(self):
        return self.case_base

    def get_weights(self):
        return self.weights

    # SET FUNCTION

    def set_function(self, column, function, matrix: pd.DataFrame):
        # if matrix is empty:
        if not matrix.empty:
            self.similarity_matrix[column] = matrix
        self.similarity_funct[column] = function

    def set_weight(self, column, weight):
        self.weights[column] = weight

    def set_weights(self, weights, autofill: bool=False):
        if autofill:
            for column in self.case_base.columns:
                if column not in weights:
                    weights[column] = 1
        self.weights = weights

    # ADD FUNCTION

    def add_case(self, case):
        if type(case) == pd.Series:
            case = pd.DataFrame(case).T

        bases_id = len(self.case_base)
        case.insert(0, 'case_id', [f"ID-{i+bases_id}" for i in range(len(case))])
        self.case_base = pd.concat([self.case_base, case])

    # MODIFY FUNCTION

    def modify_weight(self, column, weight):
        self.weights[column] = weight

    # REMOVE FUNCTION

    def remove_case(self, case_id):
        self.case_base = self.case_base[self.case_base['case_id'] != case_id]

    def remove_cases(self):
        self.case_base = pd.DataFrame(columns=self.case_base.columns)

    def remove_function(self, column):
        self.similarity_funct[column] = None

    def remove_weight(self, column):
        self.weights[column] = 0


