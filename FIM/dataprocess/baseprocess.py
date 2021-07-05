import pandas as pd

def bin_classification(data):
    for index, row in data.iteritems():
        if data[index].max()>1:
            #print(index)
            count0=0
            count1=0
            for indexrow, row in data.iterrows():
                #print(row)
                if row[index]<data[index].max()/2:
                    row[index]=0
                    count0=count0+1
                else:
                    row[index]=1
                    count1=count1+1
            #print(count0,count1)
        else:
            #print(index)
            count0=0
            count1=0
            for indexrow, row in data.iterrows():
                #print(row)
                if row[index]==0:
                    count0=count0+1
                else:
                    count1=count1+1
            #print(count0,count1)
    return data