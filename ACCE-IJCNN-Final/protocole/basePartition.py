import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics

#partitions= a set of base partitions
def writeBasePartition(fileName,partitions):
    f = open(fileName, "w")
    for p in partitions:
        f.write(str(p)+ "\n")
    f.close()
    return

#write the stats of all the base partition sets
def writeBasePartitionAnalysis(fileName,allBasePartARI,allBasePartK):
    #print(allBasePartARI)
    f = open(fileName, "w")
    f.write("Analysis of all the base partition"+"\n")
    f.write("Total number of base partition:"+str(len(allBasePartARI))+"\n\n")
    for i in range(0,len(allBasePartARI)):
        bpARI=allBasePartARI[i]
        ks=allBasePartK[i]
        mini=str(round(min(bpARI),3))
        maxi=str(round(max(bpARI),3))
        #for each BP sets write their ARI's stats
        f.write("Base partition "+str(i)+": Mean="+str(round(np.mean(bpARI),3))+", Median="+str(round(np.median(bpARI),3))+
                ",  Min="+mini+", Max="+maxi+
                ", Standard Deviation="+str(round(np.std(bpARI),3))+ "\n\n")

        min_value = min(bpARI)
        min_index = bpARI.index(min_value)
        res=[i for i, x in enumerate(bpARI) if x == min_value]
        kmin=[]
        for j in range (0,len(res)):
        	if ks[res[j]] not in kmin:
        		kmin.append(ks[res[j]])
        f.write("K inducing the minimum ("+mini+"): "+str(kmin)+" \n")


        max_value = max(bpARI)
        max_index = bpARI.index(max_value)
        res=[i for i, x in enumerate(bpARI) if x == max_value]
        kmax=[]
        for j in range (0,len(res)):
        	if ks[res[j]] not in kmax:
        		kmax.append(ks[res[j]])
        f.write("K inducing the maximum ("+maxi+"): "+str(kmax)+" \n\n")

        #f.write(str(bpARI)+"\n\n")
        #f.write(str(ks)+"\n\n")
    f.close()
    return

#write the stats of all the base partition sets
def writeBasePartitionAnalysis_C(fileName,allBasePartARI,allBasePartK,allPerBP):
    #print(allBasePartARI)
    f = open(fileName, "w")
    f.write("Analysis of all the base partition"+"\n")
    f.write("Total number of base partition:"+str(len(allBasePartARI))+"\n\n")
    for i in range(0,len(allBasePartARI)):
        bpARI=allBasePartARI[i]
        ks=allBasePartK[i]
        per=allPerBP[i]
        mini=str(round(min(bpARI),3))
        maxi=str(round(max(bpARI),3))
        #for each BP sets write their ARI's stats
        f.write("Base partition "+str(i)+": Mean="+str(round(np.mean(bpARI),3))+", Median="+str(round(np.median(bpARI),3))+
        ",  Min="+mini+", Max="+maxi+
        ", Standard Deviation="+str(round(np.std(bpARI),3))+ "\n\n")

        #for each BP sets write their conresponding percentae
        f.write("Verified constraints in BP"+str(i)+": Mean="+str(round(np.mean(per),3))+", Median="+str(round(np.median(per),3))+
                ",  Min="+str(round(min(per),3))+", Max="+str(round(max(per),3))+
                ", Standard Deviation="+str(round(np.std(per),3))+ "\n\n")


        min_value = min(bpARI)
        min_index = bpARI.index(min_value)
        res=[i for i, x in enumerate(bpARI) if x == min_value]
        kmin=[]
        permin=[]
        for j in range (0,len(res)):
            if ks[res[j]] not in kmin:
                kmin.append(ks[res[j]])
                permin.append(per[res[j]])
        f.write("K inducing the minimum ARI ("+mini+"): "+str(kmin)+" with constraint satisfaction percentage of: "+str(permin)+" \n")


        max_value = max(bpARI)
        max_index = bpARI.index(max_value)
        res=[i for i, x in enumerate(bpARI) if x == max_value]
        kmax=[]
        permax=[]
        for j in range (0,len(res)):
            if ks[res[j]] not in kmax:
                kmax.append(ks[res[j]])
                permax.append(per[res[j]])
        f.write("K inducing the maximum ARI ("+maxi+"): "+str(kmax)+" with constraint satisfaction percentage of: "+str(permax)+" \n\n")

    f.close()
    return

def readBasePartition(partfile):
    p = [] #all base partitions
    with open(partfile, 'r') as f:
        for line in f:
            line = line.strip()
            if line != '':
                part=eval(line)
                p.append(part)
    return p

def kmeansBaseline(data,k,labels):
    allKmeansPartitions=[]
    ARIs=[]
    while(len(ARIs)<10):
    #for j in range(0,10):
        km = KMeans(n_clusters=k, init='random',n_init=10, max_iter=300)#n_init:Number of time the k-means algorithm will be run with different centroid seeds; max_iter:Maximum number of iterations
        y_km=km.fit_predict(data)
        allKmeansPartitions.append(y_km)
        #Calculer l’ARI pour chaque partition de base.
        kARI=metrics.adjusted_rand_score(labels,y_km)
        ARIs.append(round(kARI,3))
    return ARIs,allKmeansPartitions
