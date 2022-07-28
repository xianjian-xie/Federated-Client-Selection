import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.cluster import KMeans

def cluster():

    relation = [100,99,0,0,0,1,1,3,97,96]
    X = np.array(relation).reshape((-1,1))
    # X = np.array([[581083], [583499], [578592],
    #               [524191], [507754], [518057],
    #               [507411], [520569], [513909], [507350]])

    # X = np.array([[401083], [403499], [408592],
    #               [524191], [507754], [518057],
    #               [507411], [520569], [403909], [507350]])
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    label = kmeans.labels_
    print('label', label)

    # kmeans.predict([[0, 0], [12, 3]])
    outlier_idx = np.where(label==1)[0]
    outlier_idx = outlier_idx.tolist()
    print('center', outlier_idx, type(outlier_idx))

    
def z_score():
    data = [200,0,0,0,0,1,1,3,5,92]
    mean = np.mean(data)
    std = np.std(data)
    print('mean of the dataset is', mean)
    print('std. deviation is', std)
    
    threshold = 1
    outlier_idx = []
    for i in range(len(data)):
        z = (data[i]-mean)/std
        print('z is', z)
        if z > threshold:
            outlier_idx.append(i)
    print('outlier in dataset is', outlier_idx)




def main():

    z_score()

    

if __name__ == '__main__':
    main()
