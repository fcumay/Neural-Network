
import math
import numpy as np
import tensorflow as tf


class ErrorFunctionException(Exception):
    def __str__(self):
        return("Unexpected input. Please come again.")




class LSTM_utils:
    def __init__(self,alpha):
        self.first_val, self.last_val, self.last_step = 0,100,1000
        self.alpha = alpha
        self.length = 50
        self.func_list = {
            "sin": lambda x: math.sin(x),
            "cos": lambda x: math.cos(x)
        }
        self.sc = self.get_sc()
        self.set_train = self.set_test = self.set_train_general = self.set_test_general = self.train_list = None
        self.make_set()


    def get_sc(self):
        succession1 = input()
        if self.func_list.get(succession1) is not None:
            return self.func_list.get(succession1)
        else:
            raise ErrorFunctionException


    def make_set(self):
        range_of_succession = np.linspace(self.first_val, self.last_val, self.last_step)
        set = list()
        for i in range_of_succession:
            tempList = list()
            tempList.append(self.sc(i))
            set.append(tempList)

        step = int(self.alpha * len(set))
        self.set_train = set[:step]
        self.set_test = set[step:]
        self.set_train_general = self.set_train[self.length:]
        self.set_test_general = range_of_succession[step:]

        self.train_list = tf.keras.utils.timeseries_dataset_from_array(
            sequence_length=self.length,
            targets=self.set_train_general,
            batch_size=10,
            data=self.set_train,
        )





