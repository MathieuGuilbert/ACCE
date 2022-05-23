from COP_Kmeans_master.copkmeans.cop_kmeans import cop_kmeans
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from constraints import *
from CAM import *
from BaselineR import writeBasePartition
from basePartition import writeBasePartitionAnalysis
import random
from math import sqrt

#lauches the COPKmeans algorithm with the given parameters.
def lauchCOPKmeans(data,k,ml,cl,max_iter):
    clusters, centers=cop_kmeans(data,k,ml,cl,'kmpp',max_iter,1e-4)
    #print(clusters)
    if clusters==None and max_iter<2200:
        #print("rec")
        return lauchCOPKmeans(data,k,ml,cl,max_iter+500)
    return clusters


#n: number of elements in the database
#k_final: number of clusters in the consensus partition
#nRepeat: number times to repat the main loop
#constrSets: ensemble containts multiple cconstraint sets
#return a matrix ith the resulting ARIs
def baselineCOP(currentDataPath,datasetName,data,n,labels,k_final,nRepeat,constrSets):
    print("-- COPKMEANS Baseline --")
    from pythonScript import Matrices,minimalSplit
    #computation of the maximum k (2 min, 50 max)
    lim=int(sqrt(n))
    if(lim>50):
        lim=50
    if(lim<2):
        lim=2

    resMatrix=[[0 for x in range(len(constrSets))] for y in range(nRepeat)]
    partitionMatrix=[[] for y in range(nRepeat)] #matrix to make the result more readable
    copDistMat=[]
    verifMatrix=[[0 for x in range(len(constrSets))] for y in range(nRepeat)]

    #for each constrSets
    for c in range(0,len(constrSets)):
        print("- Constraint set "+str(c)+" -")
        constraints=constrSets[c]
        cl,ml=extractCLandML2D(constraints)

        results=[]; resultsARI=[]; allBasePart=[]; allBasePartARI=[]; allMinSplit=[]; allDistMat=[]; allCAM=[]; allBasePartK=[]
        #Repeat nRepeat times
        for i in range(0,nRepeat):
            generateDirectory(currentDataPath+"/COP/constrSet"+str(c)+"/CP"+str(i))
            generateDirectory(currentDataPath+"/COP/constrSet"+str(c)+"/CP"+str(i)+"/basePartitions")
            ARIs=[] #ARIs of the base partitions
            allK=[]
            failedK=[]
            basePartitions=[]
            #Generate 50 base partitions with MPCKM
            for j in range(0,50):
                km=None
                while(km==None):
                    k=random.randrange(2,lim+1,1)
                    #km = KMeans(n_clusters=k, init='random',n_init=10, max_iter=300)#n_init:Number of time the k-means algorithm will be run with different centroid seeds; max_iter:Maximum number of iterations
                    km=lauchCOPKmeans(data,k,ml,cl,max_iter=300)
                    if(km==None):
                        failedK.append(k)
                allK.append(k)
                basePartitions.append(km)
                #Compute the ARI of the base partition
                kARI=metrics.adjusted_rand_score(labels,km)
                ARIs.append(kARI)
            #print("failed K:"+str(failedK))
            #print("sucessful K:"+str(allK))

            writeBasePartition(currentDataPath+"/COP/constrSet"+str(c)+"/CP"+str(i)+"/"+datasetName+"_BP_MPCKM_k=[2,"+str(lim)+"]_CS"+str(c)+"_CP"+str(i)+".txt",basePartitions)
            allBasePart.append(basePartitions)
            allBasePartARI.append(ARIs)
            allBasePartK.append(allK)

            #construction of the co-association matrix (CAM) and the distance matrix
            CAM,distMat=Matrices(basePartitions,len(basePartitions),n)
            allDistMat.append(distMat)
            allCAM.append(CAM)

            #Consensus function using Single Link
            mat = np.array(distMat)
            clustering = AgglomerativeClustering(affinity='precomputed',linkage='single',n_clusters=k_final).fit(mat)
            clustering
            finalLabels=clustering.labels_
            results.append(finalLabels)
            partitionMatrix[i].append(finalLabels)
            #compute minimal split in order to compare with MiniZinc/Gecode
            minSplit=minimalSplit(distMat,finalLabels)
            allMinSplit.append(minSplit)
            #Computation of ARI
            ARI = metrics.adjusted_rand_score(labels,finalLabels)
            resultsARI.append(ARI)
            #add the results
            resMatrix[i][c]=round(ARI,3)
            verifMatrix[i][c]=verify_constraints(finalLabels,constraints)
        copDistMat.append(allDistMat)

        #End
        currentPath=currentDataPath+"/COP/constrSet"+str(c)
        writeConsensusPartitionAnalysis(currentPath+"/consensusPartitionAnalysis.txt",allMinSplit,resultsARI)
        for r in range(0,len(allMinSplit)):
            tmpPath=currentPath+"/CP"+str(r)
            writeConsensusPartition(tmpPath+"/ConsensusPartition"+str(r)+".txt",results[r],allMinSplit[r],resultsARI[r])
            writeMatrix(tmpPath+"/DistanceMatrix"+str(r)+".txt",allDistMat[r])
            writeMatrix(tmpPath+"/CAM"+str(r)+".txt",allCAM[r])

        writeBasePartitionAnalysis(currentPath+"/CP"+str(i)+"/basePartitionAnalysis"+str(i)+".txt",allBasePartARI,allBasePartK)
        #print("constrSet "+str(c)+" done")

    print("-- End COP Baseline --")
    return resMatrix,partitionMatrix,copDistMat,verifMatrix


def copkmeansSimpleBaseline(data,k,labels,constrSets):
    allCOPKmeansPartition=[]
    allARIs=[]
    for constraints in constrSets:
        partitions=[]
        ARIs=[]
        cl,ml=extractCLandML2D(constraints)
        for j in range(0,5):
            km=None
            km = lauchCOPKmeans(data,k,ml,cl,max_iter=300)
            if(km!=None):
                partitions.append(km)
                kARI=metrics.adjusted_rand_score(labels,km)
                ARIs.append(round(kARI,3))
        allCOPKmeansPartition.append(partitions)
        if(ARIs!=[]):
            allARIs.append(ARIs)

    return allARIs
