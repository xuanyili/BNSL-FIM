class BN_Distance(object):
    def __init__(self):
        pass

class Hamming_Dis(BN_Distance):
    def __init__(self):
        super(Hamming_Dis, self).__init__()

    def distance(self, model1, model2):
        rev = 0
        sam = 0
        dis = 0
        model1_len = len(model1)
        model2_len = len(model2)
        for itemset in model1:
            itemset1 = (itemset[1], itemset[0])
            if itemset in model2:
                #print(itemset)
                sam += 1
            elif itemset1 in model2:
                rev += 1
        dis = model1_len + model2_len - 2*sam - rev
        return dis