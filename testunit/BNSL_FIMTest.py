import sys
import pandas as pd
sys.path.append(sys.path[0]+"/..")
from FIM.dataprocess import BN_Sample_Bif
from FIM.bn import BNSLFIM_Estimate, Hamming_Dis
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import BDeuScore as bds
from pgmpy.models import BayesianModel
import pyAgrum as gum

def pgmpy2agrum(model):
    arc_str = [x[0] + '->' + x[1] for x in model.edges()]
    bn = gum.fastBN(';'.join(arc_str))
    return bn

class BNSL_FIMTest(BNSLFIM_Estimate):
    def __init__(self, sample_data, target, sample_size):
        self.finalresult = pd.DataFrame(columns=("sample_size","bnsl_dis","bdeu_dis"))
        sample = BN_Sample_Bif(sample_data, sample_size)
        self.data = sample.sample2csv(target)
        self.data = pd.read_csv(target)
        self.sample_size = sample_size
        super(BNSL_FIMTest, self).__init__(self.data)
        bn = sample.sample2bn()
        #gum.about()
        #bn = gum.loadBN(sample_data)
        self.bn = self.get_origin_model(bn)
    
    def get_origin_model(self, bn_o):
        names_o=bn_o.names()
        names_o = list(names_o)
        ids = [bn_o.idFromName(name) for name in names_o]
        arcs_o=bn_o.arcs()
        d_o=[]
        for (u,v) in arcs_o:
            u,v = ids.index(u), ids.index(v)
            f=(names_o[u],names_o[v])
            d_o.append(f)
        modelo = BayesianModel(d_o)
        return modelo

    def result_estimate(self):
        self.model_bnsl = self.estimate()
        hc = HillClimbSearch(self.data)
        self.model_bdeu = hc.estimate(scoring_method=bds(self.data))
        dis = Hamming_Dis()
        tmp = pd.DataFrame(columns=("sample_size","bnsl_dis","bdeu_dis"))
        tmp["sample_size"] = [self.sample_size]
        tmp["bnsl_dis"] = [dis.distance(self.bn.edges(), self.model_bnsl.edges())]
        tmp["bdeu_dis"] = [dis.distance(self.bn.edges(), self.model_bdeu.edges())]
        self.finalresult = self.finalresult.append(tmp)

    def report(self):
        return self.finalresult
    
    def ori_bn(self):
        return pgmpy2agrum(self.bn)
    
    def bnsl_bn(self):
        return pgmpy2agrum(self.model_bnsl)
    
    def bdeu_bn(self):
        return pgmpy2agrum(self.model_bdeu)

class BNSL_FIMTestAsia(BNSL_FIMTest):
    def __init__(self, sample_size):
        super(BNSL_FIMTestAsia, self).__init__("/home/lxy/code/PrivateProtect_BN/data/asia.bif", "/home/lxy/code/PrivateProtect_BN/data/asia.csv", sample_size)
        self.min_s = 0.75
        self.min_c = 0.95
    
class BNSL_FIMTestSachs(BNSL_FIMTest):
    def __init__(self, sample_size):
        super(BNSL_FIMTestSachs, self).__init__("/home/lxy/code/PrivateProtect_BN/data/sachs.bif", "/home/lxy/code/PrivateProtect_BN/data/sachs.csv", sample_size)
        self.min_s = 0.6
        self.min_c = 0.9

class BNSL_FIMTestAlarm(BNSL_FIMTest):
    def __init__(self, sample_size):
        super(BNSL_FIMTestAlarm, self).__init__("/home/lxy/code/PrivateProtect_BN/data/alarm.bif", "/home/lxy/code/PrivateProtect_BN/data/alarm.csv", sample_size)
        self.min_s = 0.95
        self.min_c = 0.99

class BNSL_FIMTestInsurance(BNSL_FIMTest):
    def __init__(self, sample_size):
        super(BNSL_FIMTestInsurance, self).__init__("/home/lxy/code/PrivateProtect_BN/data/insurance.bif", "/home/lxy/code/PrivateProtect_BN/data/insurance.csv", sample_size)
        self.min_s = 0.75
        self.min_c = 0.98

class BNSL_FIMTestHepar2(BNSL_FIMTest):
    def __init__(self, sample_size):
        super(BNSL_FIMTestHepar2, self).__init__("/home/lxy/code/PrivateProtect_BN/data/hepar2.bif", "/home/lxy/code/PrivateProtect_BN/data/hepar2.csv", sample_size)
        self.min_s = 0.97
        self.min_c = 0.99

class BNSL_FIMTestHailfinder(BNSL_FIMTest):
    def __init__(self, sample_size):
        super(BNSL_FIMTestHailfinder, self).__init__("/home/lxy/code/PrivateProtect_BN/data/hailfinder.bif", "/home/lxy/code/PrivateProtect_BN/data/hailfinder.csv", sample_size)
        self.min_s = 0.7
        self.min_c = 0.9