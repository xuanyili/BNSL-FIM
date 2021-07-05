import numpy as np
from pgmpy.estimators import StructureScore
from scipy.special import gammaln
from math import lgamma, log
from ..dataprocess import BN_Sample_Bif, bin_classification, min_frequent_item, min_ass_subgraph, min_max_fre_itemsets, cal_black_priori, priori_structrue_PC
from pgmpy.estimators import HillClimbSearch

class BDeuScore_pt(StructureScore):
    def __init__(self, data, priori_subgraph, black_priori_subgraph, weight, equivalent_sample_size=10, **kwargs):
        """
        BNSL-FIM: add prior result in BDeu Score
        """
        self.equivalent_sample_size = equivalent_sample_size
        self.priori_subgraph = priori_subgraph
        self.black_priori_subgraph = black_priori_subgraph
        self.weight = weight
        super(BDeuScore_pt, self).__init__(data, **kwargs)

    def local_score(self, variable, parents):
        'Computes a score that measures how much a \
        given variable is "influenced" by a given list of potential parents.'

        var_states = self.state_names[variable]
        var_cardinality = len(var_states)
        state_counts = self.state_counts(variable, parents)
        num_parents_states = float(state_counts.shape[1])

        counts = np.asarray(state_counts)
        log_gamma_counts = np.zeros_like(counts, dtype=np.float_)

        alpha = self.equivalent_sample_size / num_parents_states
        beta = self.equivalent_sample_size / counts.size

        # Compute log(gamma(counts + beta))
        gammaln(counts + beta, out=log_gamma_counts)

        # Compute the log-gamma conditional sample size
        log_gamma_conds = np.sum(counts, axis=0, dtype=np.float_)
        gammaln(log_gamma_conds + alpha, out=log_gamma_conds)
        
        s = 0
        #print("parents:")
        #print(parents)
        for parent in parents:
            edge = (parent,variable)
            if edge in self.priori_subgraph:
                s += self.weight
                #print(edge, s, counts)
            elif edge in self.black_priori_subgraph:
                s -= self.weight

        score = (
            np.sum(log_gamma_counts)
            - np.sum(log_gamma_conds)
            + num_parents_states * lgamma(alpha)
            - counts.size * lgamma(beta)
            + counts.size * s
        )
        #print(lgamma(beta))
        return score

class BNSLFIM_Estimate(object):
    def __init__(self, data):
        self.data = data
        self.bindata = bin_classification(self.data)
        self.min_s = 0.7
        self.min_c = 0.9
        self.weight = 0.08
    
    def estimate(self):
        frequent_itemsets = min_frequent_item(self.bindata, self.min_s)
        ass_subgraph = min_ass_subgraph(frequent_itemsets, self.min_c)
        max_fre_itemsets = min_max_fre_itemsets(frequent_itemsets)
        priori_subgraph = priori_structrue_PC(max_fre_itemsets, ass_subgraph, self.data)
        black_priori_subgraph = cal_black_priori(max_fre_itemsets, priori_subgraph,ass_subgraph)
        hc = HillClimbSearch(self.data)
        model = hc.estimate(scoring_method=BDeuScore_pt(self.data, priori_subgraph, black_priori_subgraph, self.weight))
        return model
    
    