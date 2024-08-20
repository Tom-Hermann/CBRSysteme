import pandas as pd
import numpy as np
from enum import Enum

from sys import stderr

from fastdtw import fastdtw

from scipy.spatial.distance import euclidean

from aeon.distances import msm_distance

from scipy.fft import fft, fftfreq


# creat a something to associate color with a value: blue 0, red 1, green 2

class TimeSeriesSimFun(Enum):
    DTW = 1
    MSM = 2

class TimeModifier(Enum):
    NONE = 0
    FFT = 1
    FFTFREQ= 2



class TimeRepresentation:
    def __init__(self) -> None:
        self.is_set = False
        self.column_name = None
        pass


class TimeLine(TimeRepresentation):
    def __init__(self, starting_date, finish_date, setp, column_name) -> None:
        self.time_line: pd.DatetimeIndex = pd.DatetimeIndex(start=starting_date, end=finish_date, freq=setp)
        self.is_set: bool = True
        self.column_name = column_name
        pass

    def set_time_representation(self, starting_date, finish_date, setp):
        self.time_line = pd.DatetimeIndex(start=starting_date, end=finish_date, freq=setp)
        self.is_set = True


class ArbitraryRepresentation(TimeRepresentation):
    def __init__(self, column_name) -> None:
        self.is_set: bool = True
        self.column_name = column_name
        pass




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

        self.time_representation: TimeRepresentation = TimeRepresentation()


        self.time_representation_func: TimeSeriesSimFun = TimeSeriesSimFun.DTW
        self.time_tranformer: TimeModifier = TimeModifier.NONE

    def get_function_from_type(self, type):
        match type:
            case 'float64' | 'int64':
                return self.number_similarity
            case 'datetime64[ns]':
                return self.date_similarity
            case 'str':
                return self.matrix_similarity
            case 'object':
                return self.time_series_similarity
            case _:
                print("Type not recognized:", type)
                return None

    # SIMILARITY FUNCTION

    def time_series_similarity(self, x, y, name_column):
        # get similarity between two time series
        try:
            match self.time_representation_func:
                case TimeSeriesSimFun.DTW:
                    distance, _ = fastdtw(x, y, dist=euclidean)
                case TimeSeriesSimFun.MSM:
                    distance = msm_distance(x, y)
                case _:
                    print("Time series similarity not set")
                    print(self.time_representation_func)
                    raise ValueError("Time series similarity not set")

        except:
            print(f"Error while using fastdtw, x: {x}, y: {y}")
            raise ValueError("Error while using", self.time_representation_func)
        similarity = 1 / (1 + (distance))
        return similarity

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

        # print(x, y, type(x), type(y))

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

    def set_up_time_series(self, case1, case2, column):
        if self.time_representation.is_set:
            if self.time_representation.column_name != None:
                timeline1 = case1[self.time_representation.column_name]
                timeline2 = case2[self.time_representation.column_name]

                match self.time_tranformer:
                    case TimeModifier.NONE:
                        value1 = case1[column]
                        value2 = case2[column]
                    case TimeModifier.FFT:
                        value1 = fft(case1[column])
                        value2 = fft(case2[column])
                    case TimeModifier.FFTFREQ:
                        value1 = fftfreq(len(case1[column]))
                        value2 = fftfreq(len(case2[column]))
                    case _:
                        raise ValueError("Time modifier not set")

                # for each value create a tuple (time, value)
                if self.time_representation_func == TimeSeriesSimFun.DTW:
                    time_series1 = [(time, value) for time, value in zip(timeline1, value1)]
                    time_series2 = [(time, value) for time, value in zip(timeline2, value2)]
                elif self.time_representation_func == TimeSeriesSimFun.MSM:
                    time_series1 = np.array([timeline1, value1])
                    time_series2 = np.array([timeline2, value2])

                return time_series1, time_series2
            else:
                raise ValueError("Time representation set but no column name")
        else:

            raise ValueError("Time representation not set")




    def get_similarity_(self, case1: pd.Series, case2: pd.Series):
        similarity = 0

        for value1, value2, column in zip(case1, case2[1:], case1.index):
            if self.weights[column] == 0 or column == 'case_id':
                continue
            # similarity += self.weights[column] * (self.similarity_funct[column])(value1, value2, column)
            try:
                if self.similarity_funct[column].__name__ == 'time_series_similarity':
                    value1, value2 = self.set_up_time_series(case1, case2, column)

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



    def get_similarity(self, case1: pd.Series, case2: pd.Series):
        similarity = pd.Series()
        total = 0

        if case1['case_id'] == case2['case_id']:
            return pd.Series([0, case1['case_id'], case2['case_id']], index=['total', 'org_id', 'case_id'])


        for value1, value2, column in zip(case1, case2, case1.index):
            if column == 'case_id' or self.weights[column] == 0:
                similarity[column] = None
                continue
            # similarity += self.weights[column] * (self.similarity_funct[column])(value1, value2, column)
            try:
                if self.similarity_funct[column].__name__ == 'time_series_similarity':
                    value1, value2 = self.set_up_time_series(case1, case2, column)

                value = self.weights[column] * (self.similarity_funct[column])(value1, value2, column)
                similarity[column] = value
                total += value
            except Exception as e:
                print(f"Error similarity mesure for {column}. Similarity function : {self.similarity_funct[column].__name__}. Weight: {self.weights[column]}")
                similarity += 0
                raise e
        try:
            total = total / sum(self.weights.values())
            similarity["total"] = total
        except ZeroDivisionError:
            print("All weights are set to 0, no similarity can be computed", file=stderr)
            return 0

        try:
            similarity["org_id"] = case1["case_id"]
            similarity["case_id"] = case2["case_id"]
        except:
            similarity["org_id"] = "unknow_case"
            similarity["case_id"] = "unknow_case"

        return similarity


    def get_similaritys(self, case_id:str= "ID-0"):

        try:
            case = self.case_base[self.case_base['case_id'] == case_id].iloc[0]
        except:
            print("Case id not found")
            return None

        similarity_df = self.case_base.apply(lambda x: (self.get_similarity(case, x)), axis=1)

        # similarity_df = pd.DataFrame(similarity_df.to_list(), columns=['case_id', 'similarity'])

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

    def set_time_representation(self, time_representation: TimeRepresentation):
        self.time_representation = time_representation

    def set_time_series_similarity(self, time_series_sim: TimeSeriesSimFun):
        self.time_series_similarity = time_series_sim

    def set_time_modifier(self, time_modifier: TimeModifier):
        self.time_modifier = time_modifier

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

    def remove_matrix(self, column):
        self.similarity_matrix.remove(column)


