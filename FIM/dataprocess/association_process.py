from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import itertools
import pandas as pd
from pgmpy.estimators import PC

def min_frequent_item(data, min_s):
    frequent_itemsets = apriori(data, min_support=min_s, use_colnames=True)
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
    return frequent_itemsets

def min_ass_subgraph(frequent_itemsets, min_c):
    frequent_itemsets_ass=frequent_itemsets[frequent_itemsets['length']<=2]
    rules = association_rules(frequent_itemsets_ass, min_threshold=0.7)
    rules = rules[rules['confidence']>=min_c]
    #print(rules)
    ass_subgraph = []
    for index,rule in rules.iterrows():
        x=(list(rule['antecedents'])[0],list(rule['consequents'])[0])
        ass_subgraph.append(x)
    return ass_subgraph

def min_max_fre_itemsets(frequent_itemsets):
    frequent_itemsets_new=frequent_itemsets[frequent_itemsets['length']>1]
    max_fre_itemsets = []
    flag = False
    frequent_itemsets_new = frequent_itemsets_new.sort_values(by="length", ascending=False)
    for index,itemset in frequent_itemsets_new.iterrows():
        #print(itemset['itemsets'])
        for max_fre_itemset in max_fre_itemsets:
            if itemset['itemsets'].issubset(max_fre_itemset):
                flag = True
                break
        if not flag:
            max_fre_itemsets.append(itemset['itemsets'])
            #print(max_fre_itemsets)
        else:
            flag = False
    #for x in max_fre_itemsets[0]:
        #print(x)
    return max_fre_itemsets

def cal_black_priori(max_fre_itemsets,priori_subgraph,ass_subgraph):
    black_priori_subgraph = []
    for max_fre_itemset in max_fre_itemsets:
        for i in itertools.permutations(max_fre_itemset, 2):
            j = (i[1], i[0])
            #if not i in priori_subgraph and not j in priori_subgraph:
            if not i in priori_subgraph:
                if not i in ass_subgraph:
                    black_priori_subgraph.append(i)
    black_priori_subgraph = list(set(black_priori_subgraph))
    return black_priori_subgraph

def priori_structrue_PC(max_fre_itemsets, ass_subgraph, data):
    priori_subgraph = []
    for i in range(0,len(max_fre_itemsets)):
        node=[]
        df_node=pd.DataFrame()
        for x in max_fre_itemsets[i]:
            node.append(x)
        #print(node)
        df_node=data[node]
        c = PC(df_node)
        #print(df_node)
        best_model = c.estimate()
        for edge in best_model.edges():
            if not edge in priori_subgraph:
                #print(edge)
                if edge in ass_subgraph:
                    priori_subgraph.append(edge)
    return priori_subgraph