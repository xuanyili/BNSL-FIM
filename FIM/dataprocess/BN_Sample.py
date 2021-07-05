import pyAgrum as gum
import os
import pandas as pd

class BN_Sample(object):
    def __init__(self, sample_data, sample_size):
        self.sample_size = sample_size
        self.sample_data = sample_data
    
    def sample2csv(self):
        pass

class BN_Sample_Bif(BN_Sample):
    def __init__(self, sample_data, sample_size):
        super(BN_Sample_Bif, self).__init__(sample_data, sample_size)
        gum.about()
        self.bn = gum.loadBN(self.sample_data)
    
    def sample2csv(self, target):
        gum.generateCSV(self.bn, target, self.sample_size, True)
        data = pd.read_csv(target)
        return data

    def sample2bn(self):
        return self.bn