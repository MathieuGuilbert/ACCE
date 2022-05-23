#Important functions
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
import matplotlib.pyplot as plt
import random
from math import sqrt
import os
import time
import argparse
import subprocess
import sys
from minizinc import Driver, Model, Solver, default_driver, find_driver

#COPKmeans
from COP_Kmeans_master.copkmeans.cop_kmeans import cop_kmeans

#ILP code
from ConstrainedClusteringViaPostProcessing_master.models.pw_csize_ilp import run_modified_model
from ConstrainedClusteringViaPostProcessing_master.triplet_post import *

#side files
from data import *
from constraints import *
from basePartition import *
from CAM import createBaselineFolder,constrainedMatrices,readMatrix
from MiniZinc import *
from preTraitement import *
from preTraitementSpanningTree import *
from preTreatementACCE import *
from BaselineCOP import *
from BaselineR import *
from vizualization import *
from preTreatmentAlternativeILP import *

#Global variables
DataName=["halfmoon","ds2c2sc13","complex9","iris","glass","ionosphere","11","ds2k2","ds2k5"]
Data=[Halfmoon,ds2,complex9,iris,glass,ionosphere,[[0.0,3.0],[1.0,3.75],[2.0,5.50],[2.5,4.50],[2.0,1.0],[3.0,0.5],[4.0,-0.25],[5.25,-0.25],[3.25,4.50],[4.25,3.75],[5.25,3.0]],ds2,ds2]
GroundTruth=[HalfmoonLabels,ds2labels13,complex9labels,irisLabels,glassLabels,ionosphereLabels,[0,0,0,0,1,1,1,1,0,0,0],ds2labels2,ds2labels5]
TrueK=[2,13,9,3,7,2,2,2,5]
#      0 1 2 3  4 5 6 7


def read_data(datafile):
    data = []
    lenght=0
    debut = True
    with open(datafile, 'r') as f:
        for line in f:
            line = line.strip()
            if line != '':
                if debut:
                    lenght=line.split()
                    #print(lenght)
                    debut=False
                else:
                    d = [float(i) for i in line.split()]
                    data.append(d)
    return data

#read a file of labels and returns a list of int
def read_labels(labfile):
    l = []
    with open(labfile, 'r') as f:
        for line in f:
            line = line.strip()
            if line != '':
                line = line.split()
                label = int(line[0])
                l.append(label)
    return l

#r=range, number of base partition
#n=number of elements in the database
#Create the co-association matrix (CAM) and the distance matrix
def Matrices(baseP,r,n):
    #initialisation
    CAM = []
    distMat=[]
    for i in range(n):
        CAM.append([0]*n)
        distMat.append([0]*n)

    for l in range(n):
        for m in range(n):
            for i in range(r):
                if((baseP[i][l]==baseP[i][m])):
                    CAM[l][m]+=1

    for i in range(n):
        for j in range(n):
            CAM[i][j]=CAM[i][j]/(r) #Normalization
            distMat[i][j]=1-CAM[i][j]

    return CAM,distMat

#r=range, number of base partition
#n=number of elements in the database
#constraints=list of tuples with (point 1, point 2, type)
#Create the co-association matrix (CAM) and the distance matrix
def MatricesBaseline(baseP,r,n,constraints):
    #initialisation
    CAM = []
    distMat=[]
    for i in range(n):
        CAM.append([0]*n)
        distMat.append([0]*n)
    for i in range(r):
        for j in range(r):
            for l in range(n):
                for m in range(n):
                    if((baseP[i][l]==baseP[i][m]) and (baseP[j][l]==baseP[j][m])):
                        CAM[l][m]+=1
    for i in range(n):
        for j in range(n):
            CAM[i][j]=CAM[i][j]/(r*r) #mconvert everything between à and 1
    for (p1,p2,t) in constraints:
        if(t==1): #ML
            CAM[p1][p2]=0
            CAM[p2][p1]=0
        elif(t==-1): #CL
            CAM[p1][p2]=1
            CAM[p2][p1]=1
    for i in range(n):
        for j in range(n):
            #CAM is a similarity matrix, we transform it into a distance matrix
            distMat[i][j]=1-CAM[i][j]

    return CAM,distMat


#computation of minimal split between C's clusters
#C=list of clusters where C[i] is the cluster in wich the element i is located
#mat=distance matrix
def minimalSplit(mat,C):
    ms=1
    for c1 in range(len(C)):
        for c2 in range(len(C)-1): #-1 to avoid last loop
            if (c1!=c2 and C[c1]!=C[c2]) : #2 dif points in different cluster
                if mat[c1][c2]<ms:

                    ms=mat[c1][c2]
    return ms

#return the maximum elem in a list of list l
def maxList(l):
    m=0
    for sublist in l:
        tmpMax=max(sublist)
        if tmpMax>m:
            m=tmpMax
    return m

#name = a path, for exemple /constraints
#generate a new directory
def generateDirectory(name):
    #path = os.getcwd()
    directoryPath=name
    if not os.path.exists(directoryPath):
        os.makedirs(directoryPath) #generate a new directory

#Evidence Accumulation: EAC
#n: number of elements in the database
#k_final: number of clusters in the consensus partition
#nRepeat: number times to repat the main loop
#return all CAM and all DistMat
###return: Best CP, best CP's ARI, CP's ARI stats (min,max,mean,median,standard deviation), Best BP of best CP, Best BP in general
def clustEnsNP(currentDataPath,datasetName,data,n,labels,k_final,nRepeat):
    #ccomputation of the maximum k (2 min, 50 max)
    lim=int(sqrt(n))
    if(lim>50):
        lim=50
    if(lim<2):
        lim=2

    results=[]; resultsARI=[]; allBasePart=[]; allBasePartARI=[]; allMinSplit=[]; allDistMat=[]; allCAM=[]; allBasePartK=[]

    for i in range(0,nRepeat):
        basePartition=[]
        ARIs=[] #ARIs of the base partitions
        allK=[]
        #Generate 50 base partitions with kmeans
        for j in range(0,50):
            k=random.randrange(2,lim+1,1)
            allK.append(k)
            km = KMeans(n_clusters=k, init='random',n_init=10, max_iter=300)#n_init:Number of time the k-means algorithm will be run with different centroid seeds; max_iter:Maximum number of iterations
            y_km=km.fit_predict(data)
            basePartition.append(y_km)
            kARI=metrics.adjusted_rand_score(labels,y_km)
            ARIs.append(kARI)
            if(i==0 and len(data[0])==2 ): #vizualization of the base partition
                vizualizePartition(k,data,y_km,currentDataPath+"/basePartition/Vizualization/BP_"+str(i)+"_"+str(j)+"_k="+str(k)+".png")
        writeBasePartition(currentDataPath+"/basePartition/"+datasetName+"_BP_kmeans_k=[2,"+str(lim)+"]_"+str(i)+".txt",basePartition) #write the i-th base partition in a file
        allBasePart.append(basePartition)
        allBasePartARI.append(ARIs)
        allBasePartK.append(allK)

        #construction of the co-association matrix (CAM) and the distance matrix
        #All numbers are between 0 and 1 included
        CAM,distMat=Matrices(basePartition,len(basePartition),n)
        allDistMat.append(distMat)
        allCAM.append(CAM)

        #Consensus function using Single Link
        mat = np.array(distMat)
        #option affinity='precomputed' if using distance matrix
        clustering = AgglomerativeClustering(affinity='precomputed',linkage='single',n_clusters=k_final).fit(mat)
        clustering
        finalLabels=clustering.labels_
        results.append(finalLabels)
        vizualizePartition(k_final,data,finalLabels,currentDataPath+"/BaselineU/Vizualization/partitionU_"+str(i)+".png")
        #compute minimal split in order to compare with MiniZinc
        minSplit=minimalSplit(distMat,finalLabels)
        allMinSplit.append(minSplit)
        #Computation of ARI
        ARI = metrics.adjusted_rand_score(labels,finalLabels)
        resultsARI.append(ARI)

    createBaselineFolder(currentDataPath+"/BaselineU",allDistMat,allCAM,results,allMinSplit,resultsARI) #Exemple: createBaselineFolder(currentDataPath+"/BaselineU",[distanceMatHalfMoon],[distanceMatHalfMoon],[HalfmoonLabels],[0.9],[0.7])
    writeBasePartitionAnalysis(currentDataPath+"/basePartition/basePartitionAnalysis.txt",allBasePartARI,allBasePartK)
    #print("results clustEnsNP:"+str(results))
    return allCAM,allDistMat,results,resultsARI, allBasePartARI,allBasePart

#take a set of constr set, a set of CAM and a set of dist matrix and compute same res as previous function (CAM,distmat,Consensus partitions and analysis)
#return a matrix with the ARIs
def constrainedBaseline(currentDataPath,k_final,labels,constraintSets,CAMs,DistanceMatrices,data):
    print("--- Start Python Constrained Baseline ---\n")
    resMatrix=[[0 for x in range(len(constraintSets))] for y in range(len(DistanceMatrices))] #matrix to make the result more readable
    verifMatrix=[[] for y in range(len(DistanceMatrices))]
    partitionMatrix=[[] for y in range(len(DistanceMatrices))] #matrix to make the result more readable
    constrainedDistMat=[]

    for i in range(len(constraintSets)):
        constraints=constraintSets[i]
        path=currentDataPath+"/BaselineC/constrSets"+str(i)
        generateDirectory(path)
        results=[]
        resultsARI=[]
        allMinSplit=[]
        allDistMat=[]
        allCAM=[]
        for j in range(len(CAMs)):
            newCAM,newDistMat=constrainedMatrices(CAMs[j],DistanceMatrices[j],constraints)
            allDistMat.append(newDistMat)
            allCAM.append(newCAM)
            #same as previous function from here: Consensus function using Single Link
            mat = np.array(newDistMat)
            clustering = AgglomerativeClustering(affinity='precomputed',linkage='single',n_clusters=k_final).fit(mat)
            clustering
            finalLabels=clustering.labels_
            results.append(finalLabels)
            minSplit=minimalSplit(newDistMat,finalLabels)
            allMinSplit.append(minSplit)
            ARI = metrics.adjusted_rand_score(labels,finalLabels)
            resultsARI.append(ARI)
            resMatrix[j][i]=round(ARI,3)
            partitionMatrix[j].append(finalLabels)
            vizualizePartition(k_final,data,finalLabels,currentDataPath+"/BaselineC/Vizualization/partitionC_CP"+str(j)+"_CS"+str(i)+".png")
            #verifying constraints
            ver=verify_constraints(finalLabels,constraints)
            verifMatrix[j].append(ver)
        constrainedDistMat.append(allDistMat)

        createBaselineFolder(path,allDistMat,allCAM,results,allMinSplit,resultsARI)
    print("--- End Python Constrained Baseline ---\n")
    return resMatrix,partitionMatrix,constrainedDistMat,verifMatrix

#--- Minizinc ----
#extract a list from a string of format "[int,...,int]"
def exctractList(str):
    toWrite=False
    l=[]
    elem=""
    for i in str:
      if(not toWrite):
        if(i=="["):
          toWrite=True
      else:
        if(i=="]"):
          l.append(int(elem))
          toWrite=False
        else:
          if(i==","):
              l.append(int(elem))
              elem=""
          if(i!=" " and i!=","):
              elem+=i
    return l

#Write the results of a Minizinc execution in a file
def writeResMZN(resultPath,result,elapse,ari,rev):
    f = open(resultPath, "w")
    f.write("Minizinc Results:")
    f.write("\n "+str(result))
    f.write("\n\n Obtained in "+str(elapse))
    f.write("\n\n ARI: "+str(ari))
    if(rev!=[]):
        f.write("\n\n Reverted order: "+str(rev))
    f.close()
    return

#Not used
def writeAggreg(path,newDM,newCL,corresp):
    f = open(path, "w")
    f.write(str(newDM)+ "\n")
    f.write(str(newCL)+ "\n")
    f.write(str(corresp)+ "\n")
    f.close()
    return

#Function from the internship, not used anymore
#lauches a minizinc process using a given mzn file and a dzn file
#write the results at resultPath (partition, ARI and elapsed)
#return ari
def launchMZNcrit(mznPath,dznPath,resultPath,labels):
    print("lauch mzn ")

    # Create a MiniZinc model
    M1 = Model(mznPath)
    M1.add_file(dznPath)

    from minizinc import Instance
    start = time.time()
    # Transform Model into a instance
    gecode = Solver.lookup("gecode")
    instance1=Instance(gecode,M1)
    # Solve the instance
    result = instance1.solve()

    end = time.time()
    elapsed = end - start
    print(f'Temps d\'exécution : {elapsed:.2}s')

    print("Result: "+str(result))
    #transform string result in a list
    part=exctractList(str(result))
    ari=metrics.adjusted_rand_score(labels,part)
    print("ARI of the Minizinc result="+str(ari))

    writeResMZN(resultPath,result,elapsed,ari,[])
    print("finish mzn")
    return part,ari


# ---- MAIN ----
#datasetNumber= an int corresponding to the id of the dataset
#nConsensusPartition= number of consensus partition to generate
#nConstrSet= number of constraintSets to generate
def main(datasetNumber,nConsensusPartition,nConstrSet,options):
    print("options: "+str(options))
    subprocess.check_call(["make"], cwd="/home/mathieu/Documents/Gecode/CP4CC-MG/CP4CC-MG")

    rootPath = os.getcwd()
    mainPath=rootPath+"/Architecture"
    generateDirectory(mainPath)
    #for i in range(0,len(DataName)):
    for i in [int(datasetNumber)]:
        print("Dataset "+DataName[i])
        print("Number of elements in the dataset: "+str(len(GroundTruth[i])))
        print("Number of clusters: "+str(TrueK[i]))
        print("Number of Consensus Partition: "+str(nConsensusPartition))
        print("Number of Constraints Set: "+str(nConstrSet)+" \n")
        currentDataPath=mainPath+"/"+DataName[i]
        generateDirectory(currentDataPath)
        generateDirectory(currentDataPath+"/basePartition")
        generateDirectory(currentDataPath+"/basePartition/Vizualization")
        generateDirectory(currentDataPath+"/constraints")
        generateDirectory(currentDataPath+"/constraints/Triplet")
        generateDirectory(currentDataPath+"/BaselineU")
        generateDirectory(currentDataPath+"/BaselineU/Vizualization")
        generateDirectory(currentDataPath+"/BaselineC")
        generateDirectory(currentDataPath+"/BaselineC/Vizualization")
        generateDirectory(currentDataPath+"/MPCKM")
        generateDirectory(currentDataPath+"/MZN")
        generateDirectory(currentDataPath+"/MZN/Vizualization")
        generateDirectory(currentDataPath+"/AllocationMatrices")
        generateDirectory(currentDataPath+"/ILP")
        generateDirectory(currentDataPath+"/ILP/Anchors")
        generateDirectory(currentDataPath+"/ILP/Successive")
        generateDirectory(currentDataPath+"/ILP/Vizualization")
        generateDirectory(currentDataPath+"/OrientedILP")
        generateDirectory(currentDataPath+"/OrientedILP/Anchors")
        generateDirectory(currentDataPath+"/copILP")
        generateDirectory(currentDataPath+"/copILP/Anchors")
        generateDirectory(currentDataPath+"/copILP/Vizualization")
        generateDirectory(currentDataPath+"/Anchors")
        generateDirectory(currentDataPath+"/Gecode")
        generateDirectory(currentDataPath+"/Gecode/Aggr")

        vizualizePartition(TrueK[i],Data[i],GroundTruth[i],currentDataPath+"/"+str(DataName[i])+"_Groundtruth.png")

        infTriplets=[]
        if('tripletSimple' in options):
            constrTripletSets=createTripletConstraints(nConstrSet,100,Data[i],computeDistMat(range(len(Data[i])),Data[i]),GroundTruth[i],currentDataPath+"/constraints/Triplet/"+DataName[i]+"_Triplet")
            #print(constrTripletSets)

        #Kmeans results
        KmeansARIs,KmeansPartitions=kmeansBaseline(Data[i],TrueK[i],GroundTruth[i])

        #Evidence Acummulation (EAC) Baseline unconstrained
        allCAM,allDistMat,partitionsU,partitionsU_ARI,allBasePartARI,allBasePart=clustEnsNP(currentDataPath,DataName[i],Data[i],len(Data[i]),GroundTruth[i],TrueK[i],nConsensusPartition)

        n=len(Data[i])
        Nr=nConstrSet #5 #Nr = number of constraint set to generate
        #Nc=int(n*5/100) #Nc = number of constraints to generate for each constraints set
        if(DataName[i]=="11"): #FOR TESTS ONLY
            constrSets=createConstraints(Nr,2,Data[i],GroundTruth[i],currentDataPath+"/constraints/"+DataName[i]+"Constraints")
        else:
            constrSets=createConstraints(Nr,50,Data[i],GroundTruth[i],currentDataPath+"/constraints/"+DataName[i]+"Constraints")
            #constrSets=createMLs(Nr,50,Data[i],GroundTruth[i],currentDataPath+"/constraints/"+DataName[i]+"Constraints")
            #constrSets=createCLs(Nr,50,Data[i],GroundTruth[i],currentDataPath+"/constraints/"+DataName[i]+"Constraints")

        if('c' in options): #constrained Baseline
            partitionsC_ARI,partitionsC,constrainedDistMat,partitionsC_Ver=constrainedBaseline(currentDataPath,TrueK[i],GroundTruth[i],constrSets,allCAM,allDistMat,Data[i])

        if('cop' in options): #COPKMeans baseline
            COPKmeansARI=copkmeansSimpleBaseline(Data[i],TrueK[i],GroundTruth[i],constrSets)
            partitionsCOP_ARI,partitionsCOP,copDistMat,partitionsCOP_Ver=baselineCOP(currentDataPath,DataName[i],Data[i],len(GroundTruth[i]),GroundTruth[i],TrueK[i],nConsensusPartition,constrSets)

        if('mpckm' in options): #MPCKM baseline
            writeDataForR(Data[i],currentDataPath+"/"+str(DataName[i])+".txt")
            partitionsMPCKM_ARI=baselineMCPKM(currentDataPath,DataName[i],Data[i],len(GroundTruth[i]),GroundTruth[i],TrueK[i],nConsensusPartition,constrSets)

        #Matrices to make the result more readable
        gecodeResMatrix=[[0 for x in range(nConstrSet)] for y in range(nConsensusPartition)]
        gecodeAggrMatrix=[[0 for x in range(nConstrSet)] for y in range(nConsensusPartition)]

        # --Article ILP Kmeans protocole --
        ILPkm25=[[0 for x in range(nConstrSet)] for y in range(nConsensusPartition)]
        ILPkm25_Ver=[[0 for x in range(nConstrSet)] for y in range(nConsensusPartition)]
        ILPkm10=[[0 for x in range(nConstrSet)] for y in range(nConsensusPartition)]
        ILPkm10_Ver=[[0 for x in range(nConstrSet)] for y in range(nConsensusPartition)]
        ILPkm5=[[0 for x in range(nConstrSet)] for y in range(nConsensusPartition)]
        ILPkm5_Ver=[[0 for x in range(nConstrSet)] for y in range(nConsensusPartition)]

        # --Article ILP cop protocole --
        ILPcop25=[[0 for x in range(nConstrSet)] for y in range(nConsensusPartition)]
        ILPcop25_Ver=[[0 for x in range(nConstrSet)] for y in range(nConsensusPartition)]
        ILPcop10=[[0 for x in range(nConstrSet)] for y in range(nConsensusPartition)]
        ILPcop10_Ver=[[0 for x in range(nConstrSet)] for y in range(nConsensusPartition)]
        ILPcop5=[[0 for x in range(nConstrSet)] for y in range(nConsensusPartition)]
        ILPcop5_Ver=[[0 for x in range(nConstrSet)] for y in range(nConsensusPartition)]

        #other tests
        ILPalt=[[0 for x in range(nConstrSet)] for y in range(nConsensusPartition)]
        ILPalt_Ver=[[0 for x in range(nConstrSet)] for y in range(nConsensusPartition)]
        ILPCopAlt=[[0 for x in range(nConstrSet)] for y in range(nConsensusPartition)]
        ILPCopAlt_Ver=[[0 for x in range(nConstrSet)] for y in range(nConsensusPartition)]

        cILP=[[0 for x in range(nConstrSet)] for y in range(nConsensusPartition)]
        cILP_Ver=[[0 for x in range(nConstrSet)] for y in range(nConsensusPartition)]

        ILPTripletMatrix=[[0 for x in range(nConstrSet)] for y in range(nConsensusPartition)]
        ILPTripletMatrix_Ver=[[0 for x in range(nConstrSet)] for y in range(nConsensusPartition)]
        KmeansTripletMatrix_Ver=[[0 for x in range(nConstrSet)] for y in range(nConsensusPartition)]
        ILPTripletInfMatrix=[0 for y in range(nConsensusPartition)]
        ILPTripletInfMatrix_Ver=[0 for y in range(nConsensusPartition)]
        KmeansTripletInfMatrix_Ver=[]

        crit1Matrix=[[0 for x in range(nConstrSet)] for y in range(nConsensusPartition)]
        crit2Matrix=[[0 for x in range(nConstrSet)] for y in range(nConsensusPartition)]
        crit2MatrixEucl=[[0 for x in range(nConstrSet)] for y in range(nConsensusPartition)]
        crit2MatrixEucl_Ver=[[0 for x in range(nConstrSet)] for y in range(nConsensusPartition)]
        crit3Matrix=[[0 for x in range(nConstrSet)] for y in range(nConsensusPartition)]
        crit3Matrix_Ver=[[0 for x in range(nConstrSet)] for y in range(nConsensusPartition)]

        EuclDistMat=computeDistMat([r for r in range(len(Data[i]))],Data[i])

        for d in range(len(allDistMat)):
            distMat=allDistMat[d]
            H=int(len(Data[i])/5)
            H=50
            div1=4
            div2=10

            if('ILP25' in options):
                MatAlloc_km25=creatEuclMatAlloc_AbC(Data[i],TrueK[i],d,currentDataPath+"/Anchors",partitionsU[d],4,"Kmeans25") #25 percent
                writeAllocMatrix(MatAlloc_km25,currentDataPath+"/AllocationMatrices/AllocKmeans25"+"_"+str(d)+".txt")
                if('tripletInf' in options):
                    ILP25TripletInfMatrix=[0 for y in range(nConsensusPartition)]
                    ILP25TripletInfMatrix_Ver=[0 for y in range(nConsensusPartition)]
            if('ILP10' in options):
                MatAlloc_km10=creatEuclMatAlloc_AbC(Data[i],TrueK[i],d,currentDataPath+"/Anchors",partitionsU[d],10,"Kmeans10") #10 percent
                writeAllocMatrix(MatAlloc_km10,currentDataPath+"/AllocationMatrices/AllocKmeans10"+"_"+str(d)+".txt")
                if('tripletInf' in options):
                    ILP10TripletInfMatrix=[0 for y in range(nConsensusPartition)]
                    ILP10TripletInfMatrix_Ver=[0 for y in range(nConsensusPartition)]
            if('ILP5' in options):
                MatAlloc_km5=creatEuclMatAlloc_AbC(Data[i],TrueK[i],d,currentDataPath+"/Anchors",partitionsU[d],20,"Kmeans5") #5 percent
                writeAllocMatrix(MatAlloc_km5,currentDataPath+"/AllocationMatrices/AllocKmeans5"+"_"+str(d)+".txt")
                if('tripletInf' in options):
                    ILP5TripletInfMatrix=[0 for y in range(nConsensusPartition)]
                    ILP5TripletInfMatrix_Ver=[0 for y in range(nConsensusPartition)]

            if('aILP' in options):
                MatAlloc_AltAnchors,anchors=createAltMatAlloc(Data[i],TrueK[i],d,currentDataPath+"/Anchors",partitionsU[d],"Kmeans",EuclDistMat)
                anchorDic=createAnchorDic(Data[i],EuclDistMat,anchors)
                #print("Anchor dictionnary : "+str(anchorDic))
                writeAllocMatrix(MatAlloc_AltAnchors,currentDataPath+"/AllocationMatrices/AltAllocKmeans"+"_"+str(d)+".txt")

            if('tripletInf' in options):
                infTripletSet=createInformativeTripletConstraints(Data[i],computeDistMat(range(len(Data[i])),Data[i]),partitionsU[d],GroundTruth[i],currentDataPath+"/constraints/Triplet/"+DataName[i]+"_TripletInf")
                infTriplets.append(infTripletSet)
                #propTripletSet=propagateTripletConstraints(infTripletSet,anchors,anchorDic,EuclDistMat)
                #print("Verif prop Triplet constraints: "+str(verify_triplet_constraints(GroundTruth[i],propTripletSet))+"% out of "+str(len(propTripletSet)))
                if('ILP25' in options):
                    #ILP_res=run_modified_model(len(Data[i]),TrueK[i],MatAlloc_km25,ML,CL,GroundTruth[i])
                    #partition=writeILPRes(ILP_res,currentDataPath+"/ILP/"+str(DataName[i])+"_km25_CP"+str(d)+"_CS"+str(c)+".txt")
                    partition=run_model_triplet(len(Data[i]),TrueK[i], MatAlloc_km25, infTripletSet)
                    pARI=metrics.adjusted_rand_score(GroundTruth[i],partition)
                    print("pARI "+str(pARI))
                    ILP25TripletInfMatrix[d]=round(pARI,3)
                    ILP25TripletInfMatrix_Ver[d]=verify_triplet_constraints(partition,infTripletSet) #verify_constraints(partition,constraints)
                    vizualizePartition(TrueK[i],Data[i],partition,currentDataPath+"/ILP/Vizualization/partitionILP_tripletInf_km25_CP"+str(d)+".png")
                if('ILP10' in options):
                    #ILP_res=run_modified_model(len(Data[i]),TrueK[i],MatAlloc_km10,ML,CL,GroundTruth[i])
                    #partition=writeILPRes(ILP_res,currentDataPath+"/ILP/"+str(DataName[i])+"_km10_CP"+str(d)+"_CS"+str(c)+".txt")
                    partition=run_model_triplet(len(Data[i]),TrueK[i], MatAlloc_km10, infTripletSet)
                    pARI=metrics.adjusted_rand_score(GroundTruth[i],partition)
                    ILP10TripletInfMatrix[d]=round(pARI,3)
                    ILP10TripletInfMatrix_Ver[d]=verify_triplet_constraints(partition,infTripletSet) #verify_constraints(partition,constraints)
                    vizualizePartition(TrueK[i],Data[i],partition,currentDataPath+"/ILP/Vizualization/partitionILP_tripletInf_km10_CP"+str(d)+".png")
                if('ILP5' in options):
                    #ILP_res=run_modified_model(len(Data[i]),TrueK[i],MatAlloc_km5,ML,CL,GroundTruth[i])
                    #partition=writeILPRes(ILP_res,currentDataPath+"/ILP/"+str(DataName[i])+"_km5_CP"+str(d)+"_CS"+str(c)+".txt")
                    partition=run_model_triplet(len(Data[i]),TrueK[i], MatAlloc_km5, infTripletSet)
                    pARI=metrics.adjusted_rand_score(GroundTruth[i],partition)
                    ILP5TripletInfMatrix[d]=round(pARI,3)
                    ILP5TripletInfMatrix_Ver[d]=verify_triplet_constraints(partition,infTripletSet) #verify_constraints(partition,constraints)
                    vizualizePartition(TrueK[i],Data[i],partition,currentDataPath+"/ILP/Vizualization/partitionILP_tripletInf_km5_CP"+str(d)+".png")

                #partition=run_model_triplet(len(Data[i]),TrueK[i], MatAllocEucl_AnchorsByClust, infTripletSet) #run_modified_model(len(Data[i]),TrueK[i],MatAllocEucl_AnchorsByClust,ML,CL,GroundTruth[i]) #run_modified_model(n,k,p,ml,cl,labels) #partition=run_model_triplet(len(Data[i]),TrueK[i], MatAllocEucl_AnchorsByClust, propTripletSet)
                #pARI=metrics.adjusted_rand_score(GroundTruth[i],partition)
                #ILPTripletInfMatrix[d]=round(pARI,3)
                #ILPTripletInfMatrix_Ver[d]=verify_triplet_constraints(partition,infTripletSet)
                #vizualizePartition(TrueK[i],Data[i],partition,currentDataPath+"/ILP/Vizualization/partition_tripletInf_"+str(d)+".png")

            for c in range(0,len(constrSets)):
                constraints=constrSets[c]
                CL,ML=extractCLandML3D(constraints)
                #propConstraints=propagatePairwiseConstraints(constraints,anchors,anchorDic,EuclDistMat)
                #propConstraints=propagatePairwiseConstraintsNPC(constraints,anchors,anchorDic,EuclDistMat)
                #print("Verif prop Pariwise constraints: "+str(verify_constraints(GroundTruth[i],propConstraints))+" out of "+str(len(propConstraints)))
                #print( str(len(propConstraints))+" > "+str(len(constraints)))
                #CLprop,MLprop=extractCLandML3D(propConstraints)

                #ILP / ILP
                if('ILP25' in options):
                    if('tripletSimple' in options):
                        triplets=constrTripletSets[c]
                        partition=run_model_triplet(len(Data[i]),TrueK[i], MatAlloc_km25, triplets)
                        pARI=metrics.adjusted_rand_score(GroundTruth[i],partition)
                        ILPkm25[d][c]=round(pARI,3)
                        ILPkm25_Ver[d][c]=verify_triplet_constraints(partition,triplets)
                        vizualizePartition(TrueK[i],Data[i],partition,currentDataPath+"/ILP/Vizualization/partitionILP_triplet_km25_CP"+str(d)+"_CS"+str(c)+".png")
                    elif('tripletInf' not in options):
                        ILP_res=run_modified_model(len(Data[i]),TrueK[i],MatAlloc_km25,ML,CL,GroundTruth[i])
                        partition=writeILPRes(ILP_res,currentDataPath+"/ILP/"+str(DataName[i])+"_km25_CP"+str(d)+"_CS"+str(c)+".txt")
                        pARI=metrics.adjusted_rand_score(GroundTruth[i],partition)
                        ILPkm25[d][c]=round(pARI,3)
                        ILPkm25_Ver[d][c]=verify_constraints(partition,constraints)
                        vizualizePartition(TrueK[i],Data[i],partition,currentDataPath+"/ILP/Vizualization/partitionILP_km25_CP"+str(d)+"_CS"+str(c)+".png")
                        #--COP--
                        if('cop' in options):
                            MatAlloc_cop25=creatEuclMatAlloc_AbC(Data[i],TrueK[i],d,currentDataPath+"/copILP/Anchors",partitionsCOP[d][c],4,"COP25_CP"+str(d)+"_CS"+str(c))
                            ILP_res=run_modified_model(len(Data[i]),TrueK[i],MatAlloc_cop25,ML,CL,GroundTruth[i])
                            partition=writeILPRes(ILP_res,currentDataPath+"/OrientedILP/"+str(DataName[i])+"_cop25_CP"+str(d)+"_CS"+str(c)+".txt")
                            oARI=metrics.adjusted_rand_score(GroundTruth[i],partition)
                            ILPcop25[d][c]=round(oARI,3)
                            ILPcop25_Ver[d][c]=verify_constraints(partition,constraints)
                            #writeAllocMatrix(MatAlloc_cop25,currentDataPath+"/AllocationMatrices/AllocCop25"+"_"+str(d)+"_"+str(c)+".txt")
                if('ILP10' in options):
                    if('tripletSimple' in options):
                        triplets=constrTripletSets[c]
                        partition=run_model_triplet(len(Data[i]),TrueK[i], MatAlloc_km10, triplets)
                        pARI=metrics.adjusted_rand_score(GroundTruth[i],partition)
                        ILPkm10[d][c]=round(pARI,3)
                        ILPkm10_Ver[d][c]=verify_triplet_constraints(partition,triplets)
                        vizualizePartition(TrueK[i],Data[i],partition,currentDataPath+"/ILP/Vizualization/partitionILP_triplet_km10_CP"+str(d)+"_CS"+str(c)+".png")
                    elif('tripletInf' not in options):
                        ILP_res=run_modified_model(len(Data[i]),TrueK[i],MatAlloc_km10,ML,CL,GroundTruth[i])
                        partition=writeILPRes(ILP_res,currentDataPath+"/ILP/"+str(DataName[i])+"_km10_CP"+str(d)+"_CS"+str(c)+".txt")
                        pARI=metrics.adjusted_rand_score(GroundTruth[i],partition)
                        ILPkm10[d][c]=round(pARI,3)
                        ILPkm10_Ver[d][c]=verify_constraints(partition,constraints)
                        vizualizePartition(TrueK[i],Data[i],partition,currentDataPath+"/ILP/Vizualization/partitionILP_km10_CP"+str(d)+"_CS"+str(c)+".png")
                        #--COP--
                        if('cop' in options):
                            MatAlloc_cop10=creatEuclMatAlloc_AbC(Data[i],TrueK[i],d,currentDataPath+"/copILP/Anchors",partitionsCOP[d][c],10,"COP10_CP"+str(d)+"_CS"+str(c))
                            ILP_res=run_modified_model(len(Data[i]),TrueK[i],MatAlloc_cop10,ML,CL,GroundTruth[i])
                            partition=writeILPRes(ILP_res,currentDataPath+"/OrientedILP/"+str(DataName[i])+"_cop10_CP"+str(d)+"_CS"+str(c)+".txt")
                            oARI=metrics.adjusted_rand_score(GroundTruth[i],partition)
                            ILPcop10[d][c]=round(oARI,3)
                            ILPcop10_Ver[d][c]=verify_constraints(partition,constraints)
                            #writeAllocMatrix(MatAlloc_cop10,currentDataPath+"/AllocationMatrices/AllocCop10"+"_"+str(d)+"_"+str(c)+".txt")
                if('ILP5' in options):
                    if('tripletSimple' in options):
                        triplets=constrTripletSets[c]
                        partition=run_model_triplet(len(Data[i]),TrueK[i], MatAlloc_km5, triplets)
                        pARI=metrics.adjusted_rand_score(GroundTruth[i],partition)
                        ILPkm5[d][c]=round(pARI,3)
                        ILPkm5_Ver[d][c]=verify_triplet_constraints(partition,triplets)
                        vizualizePartition(TrueK[i],Data[i],partition,currentDataPath+"/ILP/Vizualization/partitionILP_triplet_km5_CP"+str(d)+"_CS"+str(c)+".png")
                    elif('tripletInf' not in options):
                        ILP_res=run_modified_model(len(Data[i]),TrueK[i],MatAlloc_km5,ML,CL,GroundTruth[i])
                        partition=writeILPRes(ILP_res,currentDataPath+"/ILP/"+str(DataName[i])+"_km5_CP"+str(d)+"_CS"+str(c)+".txt")
                        pARI=metrics.adjusted_rand_score(GroundTruth[i],partition)
                        ILPkm5[d][c]=round(pARI,3)
                        ILPkm5_Ver[d][c]=verify_constraints(partition,constraints)
                        vizualizePartition(TrueK[i],Data[i],partition,currentDataPath+"/ILP/Vizualization/partitionILP_km5_CP"+str(d)+"_CS"+str(c)+".png")
                        #--COP--
                        if('cop' in options):
                            MatAlloc_cop5=creatEuclMatAlloc_AbC(Data[i],TrueK[i],d,currentDataPath+"/copILP/Anchors",partitionsCOP[d][c],20,"COP5_CP"+str(d)+"_CS"+str(c))
                            ILP_res=run_modified_model(len(Data[i]),TrueK[i],MatAlloc_cop5,ML,CL,GroundTruth[i])
                            partition=writeILPRes(ILP_res,currentDataPath+"/OrientedILP/"+str(DataName[i])+"_cop5_CP"+str(d)+"_CS"+str(c)+".txt")
                            oARI=metrics.adjusted_rand_score(GroundTruth[i],partition)
                            ILPcop5[d][c]=round(oARI,3)
                            ILPcop5_Ver[d][c]=verify_constraints(partition,constraints)
                            #writeAllocMatrix(MatAlloc_cop5,currentDataPath+"/AllocationMatrices/AllocCop5"+"_"+str(d)+"_"+str(c)+".txt")


                if('aILP' in options): #Not correct
                    #ILP_res=run_modified_model(len(Data[i]),TrueK[i],MatAlloc_AltAnchors,MLprop,CLprop,GroundTruth[i]) #run_modified_model(n,k,p,ml,cl,labels)
                    ILP_res=run_modified_model(len(Data[i]),TrueK[i],MatAlloc_AltAnchors,ML,CL,GroundTruth[i])
                    partition=writeILPRes(ILP_res,currentDataPath+"/ILP/"+str(DataName[i])+"_altAbC_CP"+str(d)+"_CS"+str(c)+".txt")
                    pARI=metrics.adjusted_rand_score(GroundTruth[i],partition)
                    ILPalt[d][c]=round(pARI,3)
                    ILPalt_Ver[d][c]=verify_constraints(partition,constraints)
                    vizualizePartition(TrueK[i],Data[i],partition,currentDataPath+"/ILP/Vizualization/partition_altAbC_CP"+str(d)+"_CS"+str(c)+".png")

                if('cILP' in options):
                    cEACmatAlloc,anchors=createAltMatAlloc(Data[i],TrueK[i],d,currentDataPath+"/Anchors",partitionsC[d][c],"Kmeans",EuclDistMat)
                    ILP_res=run_modified_model(len(Data[i]),TrueK[i],cEACmatAlloc,ML,CL,GroundTruth[i])
                    partition=writeILPRes(ILP_res,currentDataPath+"/ILP/"+str(DataName[i])+"_cEAC_CP"+str(d)+"_CS"+str(c)+".txt")
                    pARI=metrics.adjusted_rand_score(GroundTruth[i],partition)
                    cILP[d][c]=round(pARI,3)
                    cILP_Ver[d][c]=verify_constraints(partition,constraints)
                    vizualizePartition(TrueK[i],Data[i],partition,currentDataPath+"/ILP/Vizualization/partition_cEAC_CP"+str(d)+"_CS"+str(c)+".png")

                if('copILP' in options):

                    copEucMatAlloc=creatEuclMatAlloc_AbC(Data[i],TrueK[i],d,currentDataPath+"/OrientedILP/Anchors",partitionsCOP[d][c],div1,"COP")
                    ILP_res=run_modified_model(len(Data[i]),TrueK[i],copEucMatAlloc,ML,CL,GroundTruth[i]) #run_modified_model(n,k,p,ml,cl,labels)
                    partition=writeILPRes(ILP_res,currentDataPath+"/OrientedILP/"+str(DataName[i])+"_COPLocalEucl_CP"+str(d)+"_CS"+str(c)+"_Div"+str(div1)+".txt")
                    oARI=metrics.adjusted_rand_score(GroundTruth[i],partition)
                    ILPCopEucMatrix[d][c]=round(oARI,3)
                    ILPCopEucMatrix_Ver[d][c]=verify_constraints(partition,constraints)

                    copEucMatAlloc2=creatEuclMatAlloc_AbC(Data[i],TrueK[i],d,currentDataPath+"/OrientedILP/Anchors",partitionsCOP[d][c],div2,"COP")
                    ILP_res=run_modified_model(len(Data[i]),TrueK[i],copEucMatAlloc,ML,CL,GroundTruth[i]) #run_modified_model(n,k,p,ml,cl,labels)
                    partition=writeILPRes(ILP_res,currentDataPath+"/OrientedILP/"+str(DataName[i])+"_COPLocalEucl2_CP"+str(d)+"_CS"+str(c)+"_Div"+str(div2)+".txt")
                    oARI=metrics.adjusted_rand_score(GroundTruth[i],partition)
                    ILPCopEucMatrix2[d][c]=round(oARI,3)
                    ILPCopEucMatrix2_Ver[d][c]=verify_constraints(partition,constraints)

                if('copILPa' in options):
                    copMatAlloc,copAnchors=createAltMatAlloc(Data[i],TrueK[i],d,currentDataPath+"/OrientedILP/Anchors",partitionsCOP[d][c],"COP",EuclDistMat)
                    ILP_res=run_modified_model(len(Data[i]),TrueK[i],copMatAlloc,ML,CL,GroundTruth[i])
                    partition=writeILPRes(ILP_res,currentDataPath+"/OrientedILP/"+str(DataName[i])+"_COPLocalEucl_CP"+str(d)+"_CS"+str(c)+"_Div"+str(div2)+".txt")
                    oARI=metrics.adjusted_rand_score(GroundTruth[i],partition)
                    ILPCopAlt[d][c]=round(oARI,3)
                    ILPCopAlt_Ver[d][c]=verify_constraints(partition,constraints)

                # Minizinc: criteria 2
                if('mzn2' in options):
                    print("mzn criterion 2 ")
                    print("original partition ARI: "+str(partitionsU_ARI[d]))
                    #print("original partition: "+str(partitionsU[d]))
                    path=currentDataPath+"/MZN/"+str(DataName[i])+"_crit2_CP"+str(d)+"_CS"+str(c)
                    writeMZNcrit2(path,constrSets[c])
                    #writeDZN_MatAlloc(path,allDistMat[d],len(Data[i]),TrueK[i],TrueK[i],MatAlloc,partitionsU[d])
                    writeDZN_MatAlloc(path,allDistMat[d],len(Data[i]),TrueK[i],TrueK[i],MatAllocEucCor,partitionsU[d])
                    MZNres,crit2ARI=launchMZNcrit(path+".mzn",path+".dzn",path+"_res.txt",GroundTruth[i])
                    crit2Matrix[d][c]=round(crit2ARI,3)
                    #vizualizePartition(TrueK[i],Data[i],MZNres,currentDataPath+"/MZN/Vizualization/partitionMZN2_Eucl_Corrected_CP"+str(d)+"_CS"+str(c)+".png")
                    #PdifG,inConstr=accordanceMoveConstr(partitionsU[d],MZNres,constrSets[c])
                    path=currentDataPath+"/MZN/"+str(DataName[i])+"_crit2_CP"+str(d)+"_CS"+str(c)+"_Eucl"
                    writeMZNcrit2(path,constrSets[c])
                    writeDZN_MatAlloc(path,allDistMat[d],len(Data[i]),TrueK[i],TrueK[i],MatAllocEucl_AnchorsByClust,partitionsU[d])
                    MZNres,crit2ARI=launchMZNcrit(path+".mzn",path+".dzn",path+"_res.txt",GroundTruth[i])
                    crit2MatrixEucl[d][c]=round(crit2ARI,3)
                    print(MZNres)
                    crit2MatrixEucl_Ver[d][c]=verify_constraints([MZNres[e]-1 for e in range(len(MZNres))],constraints)

                # Minizinc: criteria 3
                if('mzn3' in options):
                    print("mzn criterion 3")
                    print("original partition ARI: "+str(partitionsU_ARI[d]))
                    #print("original partition: "+str(partitionsU[d]))
                    path=currentDataPath+"/MZN/"+str(DataName[i])+"_crit3_CP"+str(d)+"_CS"+str(c)
                    writeMZNcrit3(path,constrSets[c])
                    #writeDZN_MatAlloc(path,allDistMat[d],len(Data[i]),TrueK[i],TrueK[i],MatAlloc,partitionsU[d])
                    writeDZN_MatAlloc(path,allDistMat[d],len(Data[i]),TrueK[i],TrueK[i],MatAllocEucCor,partitionsU[d])
                    MZNres,crit3ARI=launchMZNcrit(path+".mzn",path+".dzn",path+"_res.txt",GroundTruth[i])
                    crit3Matrix[d][c]=round(crit3ARI,3) #need to read parti from result
                    crit3Matrix_Ver[d][c]=verify_constraints([MZNres[e]-1 for e in range(len(MZNres))],constraints)
                    #vizualizePartition(TrueK[i],Data[i],MZNres,currentDataPath+"/MZN/Vizualization/partitionMZN3_Eucl_Corrected_CP"+str(d)+"_CS"+str(c)+".png")

                # Minizinc: criteria 3
                if('mzn1' in options):
                    print("mzn criterion 1")
                    print("original partition ARI: "+str(partitionsU_ARI[d]))
                    #print("original partition: "+str(partitionsU[d]))
                    path=currentDataPath+"/MZN/"+str(DataName[i])+"_crit1_CP"+str(d)+"_CS"+str(c)
                    writeMZNcrit1(path,constrSets[c])
                    #writeDZN_MatAlloc(path,allDistMat[d],len(Data[i]),TrueK[i],TrueK[i],MatAlloc,partitionsU[d])
                    writeDZN_MatAlloc(path,allDistMat[d],len(Data[i]),TrueK[i],TrueK[i],MatAllocEucl_AnchorsByClust,partitionsU[d])
                    MZNres,crit1ARI=launchMZNcrit(path+".mzn",path+".dzn",path+"_res.txt",GroundTruth[i])
                    crit1Matrix[d][c]=round(crit1ARI,3) #need to read parti from result

                #print("Dataset "+DataName[i]+", distMat"+str(d)+" and constrSet="+str(c))
                if("cpclust" in options):
                    tARI=genericGecodeProtocole(c,i,d,currentDataPath) #genericGecodeProtocole(idConstrSet,idDataset,idConsensusPartition,currentDataPath):
                    gecodeResMatrix[d][c]=tARI


            print(" _____________ End CP"+str(d)+" _____________\n")

        allResults={}
        baselineResults={}
        experimentalResults={}
        baselineVer={}
        experimentalVer={}
        pairwiseVer={}
        TripletVer={}

        partitionsB_ARI=[]
        partitionsB_Ver=[]
        for b in range(len(allBasePartARI)):
            partitionsB_ARI.append(np.mean(allBasePartARI[b]))
            #partitionsB_Ver.append(np.mean(verify_multiple_constrSets(allBasePart[b],constrSets)))
        for y in range(nConsensusPartition): partitionsB_ARI[y]=round(partitionsB_ARI[y],3)
        print("Base partitions mean ARI: "+str(partitionsB_ARI))
        allResults['BP. Km']=partitionsB_ARI
        baselineResults['BP Km']=partitionsB_ARI
        #baselineVer['Base partitions - KMeans']=partitionsB_Ver

        print("Kmeans results with k="+str(TrueK[i])+" :"+str(KmeansARIs))
        allResults["Kmeans"]=KmeansARIs
        baselineResults["Kmeans"]=KmeansARIs
        KmeansVer=[]
        KmeansTripletVer=[]
        for w in range(len(KmeansPartitions)):
            KmeansVer.append(np.mean(verify_multiple_constrSets(KmeansPartitions[w],constrSets)))
            if('tripletSimple' in options):
                KmeansTripletVer.append(np.mean(verify_multiple_tripletSets(KmeansPartitions[w],constrTripletSets)))
                #print("Kmeans triplet ver with k="+str(TrueK[i])+" :"+str(KmeansTripletVer))
            if('tripletInf' in options):
                KmeansTripletInfMatrix_Ver.append(np.mean(verify_multiple_tripletSets(KmeansPartitions[w],infTriplets))) #verify_triplet_constraints(KmeansPartitions[w],infTriplets[w]))
                #print("Kmeans triplet ver with k="+str(TrueK[i])+" :"+str(KmeansTripletInfMatrix_Ver))
        baselineVer["Kmeans"]=KmeansVer
        pairwiseVer["Kmeans"]=KmeansVer
        if('tripletSimple' in options):
            TripletVer["Kmeans"]=KmeansTripletVer
            baselineVer["Kmeans"]=KmeansTripletVer
        if('tripletInf' in options):
            TripletVer["Kmeans"]=KmeansTripletInfMatrix_Ver
            baselineVer["Kmeans"]=KmeansTripletVer

        for y in range(nConsensusPartition): partitionsU_ARI[y]=round(partitionsU_ARI[y],3)
        print("Unconstrained Baseline results: "+str(partitionsU_ARI))
        allResults['EAC']=partitionsU_ARI
        baselineResults['EAC']=partitionsU_ARI #baselineResults['Unconstrained Baseline']=partitionsU_ARI
        partitionsU_Ver=[]
        partitionsU_TripletVer=[]
        partitionsU_TripletInfVer=[]
        for u in range(len(partitionsU)):
            partitionsU_Ver.append(np.mean(verify_multiple_constrSets(partitionsU[u],constrSets)))
            if('tripletSimple' in options):
                partitionsU_TripletVer.append(np.mean(verify_multiple_tripletSets(partitionsU[u],constrTripletSets)))
            if('tripletInf' in options):
                partitionsU_TripletInfVer.append(np.mean(verify_triplet_constraints(partitionsU[u],infTriplets[u])))
        baselineVer['EAC']=partitionsU_Ver #baselineVer['Unconstrained Baseline']=partitionsU_Ver
        pairwiseVer['EAC']=partitionsU_Ver
        if('tripletSimple' in options):
            TripletVer['EAC']=partitionsU_TripletVer
            baselineVer['EAC']=partitionsU_TripletVer
        if('tripletInf' in options):
            TripletVer['EAC']=partitionsU_TripletInfVer
            baselineVer['EAC']=partitionsU_TripletInfVer

        if('c' in options or 'og' in options):
            print("Constrained Baseline results: "+str(partitionsC_ARI))
            allResults['Constr EAC']=partitionsC_ARI
            baselineResults['Constr EAC']=partitionsC_ARI
            baselineVer['Constr EAC']=partitionsC_Ver
            pairwiseVer['Constr EAC']=partitionsC_Ver
        if('mpckm' in options):
            print("MPCKM Baseline results: "+str(partitionsMPCKM_ARI))
            allResults['MPCKM Bas.']=partitionsMPCKM_ARI
            baselineResults['MPCKM Bas.']=partitionsMPCKM_ARI

        if('cop' in options):
            print("COPKmeans results with k="+str(TrueK[i])+" : "+str(COPKmeansARI))
            if(COPKmeansARI!=[]):
                allResults["COP"]=COPKmeansARI
                baselineResults["COP"]=COPKmeansARI
                #TODO add ver

            print("COPKmeans Baseline results: "+str(partitionsCOP_ARI))
            allResults['COP EAC']=partitionsCOP_ARI
            baselineResults['COP EAC']=partitionsCOP_ARI
            baselineVer['COP EAC']=partitionsCOP_Ver
            pairwiseVer['COP EAC']=partitionsCOP_Ver

        if('copILPa' in options):
            print("ACCE-ILP alt COPKmeans "+str(div1)+": "+str(ILPCopAlt))
            allResults['ILP-COP alt']=ILPCopAlt #'COPKmeans Euclidean ILP'
            experimentalResults['ILP-COP alt']=ILPCopAlt#experimentalResults['ACCE-ILP local'+str(div1)+'-COPKmeans ']=ILPCopEucMatrix
            experimentalVer['ILP-COP alt']=ILPCopAlt_Ver
            pairwiseVer['ILP-COP alt']=ILPCopAlt_Ver

        if('ILP25' in options):
            if('tripletSimple' in options):
                print("ACCE-ILP Kmeans25 : "+str(ILPkm25))
                allResults['ILP-Km25']=ILPkm25
                experimentalResults['ILP-Km25']=ILPkm25
                experimentalVer['ILP-Km25']=ILPkm25_Ver
                TripletVer['ILP-Km25']=ILPkm25_Ver
            elif('tripletInf' in options):
                print("ACCE-ILP Kmeans25 : "+str(ILP25TripletInfMatrix))
                allResults['ILP-Km25']=ILP25TripletInfMatrix
                experimentalResults['ILP-Km25']=ILP25TripletInfMatrix
                TripletVer['ILP-Km25']=ILP25TripletInfMatrix_Ver
                experimentalVer['ILP-Km25']=ILP25TripletInfMatrix_Ver
            else:
                print("ACCE-ILP Kmeans25 : "+str(ILPkm25))
                allResults['ILP-Km25']=ILPkm25
                experimentalResults['ILP-Km25']=ILPkm25
                experimentalVer['ILP-Km25']=ILPkm25_Ver
                pairwiseVer['ILP-Km25']=ILPkm25_Ver
                if('cop' in options):
                    print("ACCE-ILP COP25 : "+str(ILPcop25))
                    allResults['ILP-COP25']=ILPcop25
                    experimentalResults['ILP-COP25']=ILPcop25
                    experimentalVer['ILP-COP25']=ILPcop25_Ver
                    pairwiseVer['ILP-COP25']=ILPcop25_Ver
        if('ILP10' in options):
            if('tripletSimple' in options):
                print("ACCE-ILP Kmeans10 : "+str(ILPkm10))
                allResults['ILP-Km10']=ILPkm10
                experimentalResults['ILP-Km10']=ILPkm10
                experimentalVer['ILP-Km10']=ILPkm10_Ver
                TripletVer['ILP-Km10']=ILPkm10_Ver
            elif('tripletInf' in options):
                print("ACCE-ILP Kmeans10 : "+str(ILP10TripletInfMatrix))
                allResults['ILP-Km10']=ILP10TripletInfMatrix
                experimentalResults['ILP-Km10']=ILP10TripletInfMatrix
                TripletVer['ILP-Km10']=ILP10TripletInfMatrix_Ver
                experimentalVer['ILP-Km10']=ILP10TripletInfMatrix_Ver
            else:
                print("ACCE-ILP Kmeans10 : "+str(ILPkm10))
                allResults['ILP-Km10']=ILPkm10
                experimentalResults['ILP-Km10']=ILPkm10
                experimentalVer['ILP-Km10']=ILPkm10_Ver
                pairwiseVer['ILP-Km10']=ILPkm10_Ver
                if('cop' in options):
                    print("ACCE-ILP COP10 : "+str(ILPcop10))
                    allResults['ILP-COP10']=ILPcop10
                    experimentalResults['ILP-COP10']=ILPcop10
                    experimentalVer['ILP-COP10']=ILPcop10_Ver
                    pairwiseVer['ILP-COP10']=ILPcop10_Ver
        if('ILP5' in options):
            if('tripletSimple' in options):
                print("ACCE-ILP Kmeans5 : "+str(ILPkm5))
                allResults['ILP-Km5']=ILPkm5
                experimentalResults['ILP-Km5']=ILPkm5
                experimentalVer['ILP-Km5']=ILPkm5_Ver
                TripletVer['ILP-Km5']=ILPkm5_Ver
            elif('tripletInf' in options):
                print("ACCE-ILP Kmeans5 : "+str(ILP5TripletInfMatrix))
                allResults['ILP-Km5']=ILP5TripletInfMatrix
                experimentalResults['ILP-Km5']=ILP5TripletInfMatrix
                TripletVer['ILP-Km5']=ILP5TripletInfMatrix_Ver
                experimentalVer['ILP-Km5']=ILP5TripletInfMatrix_Ver
            else:
                print("ACCE-ILP Kmeans5 : "+str(ILPkm5))
                allResults['ILP-Km5']=ILPkm5
                experimentalResults['ILP-Km5']=ILPkm5
                experimentalVer['ILP-Km5']=ILPkm5_Ver
                pairwiseVer['ILP-Km5']=ILPkm5_Ver
                if('cop' in options):
                    print("ACCE-ILP COP5 : "+str(ILPcop5))
                    allResults['ILP-COP5']=ILPcop5
                    experimentalResults['ILP-COP5']=ILPcop5
                    experimentalVer['ILP-COP5']=ILPcop5_Ver
                    pairwiseVer['ILP-COP5']=ILPcop5_Ver

        if('aILP' in options):
            print("ACCE-ILP alt "+str(ILPalt))
            allResults['ILP-alt']=ILPalt
            experimentalResults['ILP-alt']=ILPalt #ILP AnchorsByClust Euc
            experimentalVer['ILP-alt']=ILPalt_Ver
            pairwiseVer['ILP-alt']=ILPalt_Ver
        if('cILP' in options):
            print("cEAC-ILP "+str(cILP))
            allResults['cEAC-ILP']=cILP
            experimentalResults['cEAC-ILP']=cILP #ILP AnchorsByClust Euc
            experimentalVer['cEAC-ILP']=cILP_Ver
            pairwiseVer['cEAC-ILP']=cILP_Ver
        #if('tripletSimple' in options):
        #    print("ACCE-ILP triplet "+str(div1)+":  "+str(ILPTripletMatrix))
        #    print("ACCE-ILP triplet ver "+str(div1)+":  "+str(ILPTripletMatrix_Ver))
        #    allResults['ILP T-S']=ILPTripletMatrix
        #    TripletVer['ILP T-S']=ILPTripletMatrix_Ver
        #if('tripletInf' in options):
            #print("ACCE-ILP Informative triplet "+str(div1)+":  "+str(ILPTripletInfMatrix))
            #print("ACCE-ILP Informative triplet ver "+str(div1)+":  "+str(ILPTripletInfMatrix_Ver))
            #allResults['ILP T-I']=ILPTripletInfMatrix
            #TripletVer['ILP T-I']=ILPTripletInfMatrix_Ver
        if('mzn1' in options):
            print("Minizinc (criterion 1) results matrix: "+str(crit1Matrix))
            allResults['mzn1']=crit1Matrix
            experimentalResults['mzn1']=crit1Matrix
        if('mzn2' in options):
            print("Minizinc (criterion 2) eucl res matrix: "+str(crit2MatrixEucl))
            allResults['CP2']=crit2MatrixEucl
            experimentalResults['CP2']=crit2MatrixEucl #MZN2 eucl
            experimentalVer['CP2']=crit2MatrixEucl_Ver
            pairwiseVer['CP2']=crit2MatrixEucl_Ver
        if('mzn3' in options):
            print("Minizinc (criterion 3) results matrix: "+str(crit3Matrix))
            allResults['CP3']=crit3Matrix
            experimentalResults['CP3']=crit3Matrix
            experimentalVer['CP3']=crit3Matrix_Ver
            pairwiseVer['CP3']=crit3Matrix_Ver
        if('cpclust' in options):
            print("CPcluster results matrix: "+str(gecodeResMatrix))
            allResults['cpclust']=gecodeResMatrix
            experimentalResults['cpclust']=gecodeResMatrix

        order=['BP. Km', 'Kmeans', 'Kmeans Inf', 'EAC', 'EAC inf', 'Constr EAC',
        'ILP-Km4', 'ILP-Km25', 'ILP-Km10', 'ILP-Km5', 'ILP-old', 'ILP-alt', 'ILP', 'cEAC-ILP',
         'ILP Tripl', 'ILP T-S', 'ILP T-I', 'CP2', 'CP3', 'COP', 'COP EAC', 'ILP-COP4', 'ILP-COP25', 'ILP-COP10', 'ILP-COP5','ILP-COP alt']
        colors={'BP. Km':'cyan', 'Kmeans':'cyan', 'Kmeans Inf':'cyan', 'EAC':'blue', 'EAC inf':'blue', 'Constr EAC':'blue',
         'ILP-Km4':'red', 'ILP-Km25':'red', 'ILP-Km10':'red', 'ILP-Km5':'red', 'ILP-old':'red', 'ILP-alt':'red', 'ILP':'red', 'cEAC-ILP':'red',
          'ILP Tripl':'red', 'ILP T-S':'red', 'ILP T-I':'red',
          'CP2':'red', 'CP3':'red',
          'COP':'cyan', 'COP EAC':'blue', 'ILP-COP4':'red', 'ILP-COP25':'red', 'ILP-COP10':'red', 'ILP-COP5':'red', 'ILP-COP alt':'red'}

        #writeAllResults(allResults,currentDataPath+"/"+str(DataName[i]),options)
        #baselineStats,experimentalStats=writeSeparatedResults(baselineResults,experimentalResults,currentDataPath+"/"+str(DataName[i]))
        baselineStats,experimentalStats=writeSeparatedResultsQuartiles(baselineResults,experimentalResults,currentDataPath+"/"+str(DataName[i]))
        writeBarChart(baselineStats,experimentalStats,1.05,currentDataPath+"/"+str(DataName[i]),str(DataName[i]),"ARI")

        #baselineStatsVer,experimentalStatsVer=writeSeparatedResults(baselineVer,experimentalVer,currentDataPath+"/"+str(DataName[i])+"_VerifiedConstraints")
        #baselineStatsVer,experimentalStatsVer=writeSeparatedResultsQuartiles(baselineVer,experimentalVer,currentDataPath+"/"+str(DataName[i])+"_VerifiedConstraints")
        #writeBarChart(baselineStatsVer,experimentalStatsVer,105,currentDataPath+"/"+str(DataName[i])+"_VerifiedConstraints",str(DataName[i]),"VerifiedConstraints")

        vizBoxPlot(order,colors,allResults,currentDataPath+"/"+str(DataName[i])+"_ARI",str(DataName[i])+" ARI",1.05)
        #vizBoxPlot(order,colors,experimentalVer,currentDataPath+"/"+str(DataName[i])+"_PairwiseVer",str(DataName[i])+" Ver",105)
        if('tripletSimple' in options or 'tripletInf' in options):
            vizBoxPlot(order,colors,TripletVer,currentDataPath+"/"+str(DataName[i])+"_TripletVer",str(DataName[i])+" Triplet satisfaction",105)
        else:
            vizBoxPlot(order,colors,pairwiseVer,currentDataPath+"/"+str(DataName[i])+"_PairwiseVer",str(DataName[i])+" Ver",105)

#takes in parameter a list (or a list of list) and returns a list organised as follows:
# Mean, Median, Min, Max, Standard Deviation
def createListStat(l):
    List_flat = []
    if(type(l[0])==list):
        for a in range(len(l)):
          for b in range (len(l[a])):
            List_flat.append(l[a][b])
    else:
        List_flat=l
    return([round(np.mean(l),3),round(np.median(l),3),min(List_flat),max(List_flat),round(np.std(List_flat),3)])

#takes in parameter a list (or a list of list) and returns a list organised as follows:
# Mean, Median, Min, Max, Q1, Q3, Standard Deviation
def createListStatQuartiles(l):
    List_flat = []
    if(type(l[0])==list):
        for a in range(len(l)):
          for b in range (len(l[a])):
            List_flat.append(l[a][b])
    else:
        List_flat=l
    return([round(np.mean(l),3),round(np.median(l),3),min(List_flat),max(List_flat),round(np.percentile(List_flat,25),3),round(np.percentile(List_flat,75),3),round(np.std(List_flat),3)])

#Write a given allocation matrix in a file
def writeAllocMatrix(allocMatrix,path):
    h = open(path, "w")
    h.write(str(allocMatrix))
    h.close()
    return

def writeBarChart(baselineStats,experimentalStats,ymax,path,name,type):
    order="BP Km., Kmeans, EAC, Constr EAC, COP, COP EAC, ILP-COP4, ILP-Km4, CP2"
    ynames=set()
    for j in baselineStats:
        ynames.add(j)
    if(experimentalStats!=[]):
        for j in experimentalStats:
            ynames.add(j)
    f = open(path+"_ResultsPlot.txt", "w")
    f.write("\\begin{tikzpicture}[scale=0.57]\n")
    f.write("    \\begin{axis}[\n")
    f.write("        title="+name+" "+type+", \n")
    f.write("        symbolic x coords={"+order+"},\n") #{U, C, ILP,ILP2},\n")
    #f.write("        symbolic x coords="+str(ynames).replace("'", "")+",\n") #{U, C, ILP,ILP2},\n")
    f.write("        xtick=data, ymin=0,ymax="+str(ymax)+",bar width=15pt,height=6cm ,width=20cm \n")
    f.write("      ]\n")
    f.write("        \\addplot[ybar,fill=blue] coordinates {\n")
    for i in baselineStats:
        f.write("            ("+str(i)+", "+str(baselineStats[i][0])+")\n")
    if(experimentalStats!=[]):
        for j in experimentalStats:
            f.write("            ("+str(j)+", "+str(0.0)+")\n") #to initiate the visuals
    f.write("      };\n")

    f.write("        \\addplot[ybar,fill=red] coordinates {\n")
    if(experimentalStats!=[]):
        for j in experimentalStats:
            f.write("            ("+str(j)+", "+str(experimentalStats[j][0])+")\n")
    f.write("      };\n")
    f.write("   \\end{axis}\n")
    f.write("\\end{tikzpicture}\n")
    f.close()
    return

def writeResultsLatex(allStats,path):
    best=max([allStats[i][0] for i in allStats])
    h = open(path, "w")
    h.write("\\begin{table}[h] \n")
    h.write("\\begin{tabular}{l|l|l|l|l|l} \n")
    h.write("Method                 & Mean & Median & Min & Max & Standard deviation \\\\ \hline\hline \n")
    for i in allStats:
        currentStats=allStats[i]
        if(currentStats[0]<best-0.01):
            h.write(" "+str(i)+" & "+str(currentStats[0])+"  & "+str(currentStats[1])+"  & "+str(currentStats[2])+"  & "+str(currentStats[3])+"  & "+str(currentStats[4])+"   \\\\ \hline \n")
        else:
            h.write(" \\textbf{"+str(i)+"} & \\textbf{"+str(currentStats[0])+"}  & \\textbf{"+str(currentStats[1])+"}  & \\textbf{"+str(currentStats[2])+"}  & \\textbf{"+str(currentStats[3])+"}  & \\textbf{"+str(currentStats[4])+"}   \\\\ \hline \n")
    h.write("\end{tabular}\n")
    h.write("\end{table}\n")
    h.close()
    return

def writeResultsQuartilesLatex(allStats,path):
    best=max([allStats[i][0] for i in allStats])
    h = open(path, "w")
    h.write("\\begin{table}[h] \n")
    h.write("\\begin{tabular}{l|l|l|l|l|l} \n")
    h.write("Method                 & Mean & Median & Min & Max & Q1 & Q3 & Standard deviation \\\\ \hline\hline \n")
    for i in allStats:
        currentStats=allStats[i]
        if(currentStats[0]<best-0.01):
            h.write(" "+str(i)+" & "+str(currentStats[0])+"  & "+str(currentStats[1])+"  & "+str(currentStats[2])+"  & "+str(currentStats[3])+
            "  & "+str(currentStats[4])+"  & "+str(currentStats[5])+"  & "+str(currentStats[6])+"   \\\\ \hline \n")
        else:
            h.write(" \\textbf{"+str(i)+"} & \\textbf{"+str(currentStats[0])+"}  & \\textbf{"+str(currentStats[1])+"}  & \\textbf{"+str(currentStats[2])+"}  & \\textbf{"+str(currentStats[3])+
            "}  & \\textbf{"+str(currentStats[4])+"}  & \\textbf{"+str(currentStats[5])+"}  & \\textbf{"+str(currentStats[6])+"}   \\\\ \hline \n")
    h.write("\end{tabular}\n")
    h.write("\end{table}\n")
    h.close()
    return

def writeSeparatedResults(baselineResults,experimentalResults,path):
    baselineStats={}
    f = open(path+"_BaselineResults.txt", "w")
    g = open(path+"_BaselineResultsSummary.txt", "w")
    for i in baselineResults:
        f.write(str(i)+"\n")
        tmpRes=baselineResults[i]
        f.write(str(tmpRes)+"\n\n")
        g.write(str(i)+"\n")
        listStats=createListStat(tmpRes)
        baselineStats[i]=listStats
        g.write("ARI analysis: Mean="+str(listStats[0])+" ,Median="+str(listStats[1])+
            " ,Min="+str(listStats[2])+" ,Max="+str(listStats[3])+" ,Standard Deviation="+str(listStats[4])+ "\n\n")
    f.close()
    g.close()

    experimentalStats={}
    f = open(path+"_ExperimentalResults.txt", "w")
    g = open(path+"_ExperimentalResultsSummary.txt", "w")
    for i in experimentalResults:
        f.write(str(i)+"\n")
        tmpRes=experimentalResults[i]
        f.write(str(tmpRes)+"\n\n")
        g.write(str(i)+"\n")
        listStats=createListStat(tmpRes)
        experimentalStats[i]=listStats
        g.write("ARI analysis: Mean="+str(listStats[0])+" ,Median="+str(listStats[1])+
            " ,Min="+str(listStats[2])+" ,Max="+str(listStats[3])+" ,Standard Deviation="+str(listStats[4])+ "\n\n")
    f.close()
    g.close()
    #Write Latex
    writeResultsLatex(baselineStats,path+"_BaselineResLateX.txt")
    if(experimentalStats!={}):
        writeResultsLatex(experimentalStats,path+"_ExperimentalResLateX.txt")

    return baselineStats,experimentalStats

def writeSeparatedResultsQuartiles(baselineResults,experimentalResults,path):
    baselineStats={}
    f = open(path+"_BaselineResults.txt", "w")
    g = open(path+"_BaselineResultsSummary.txt", "w")
    for i in baselineResults:
        f.write(str(i)+"\n")
        tmpRes=baselineResults[i]
        f.write(str(tmpRes)+"\n\n")
        g.write(str(i)+"\n")
        listStats=createListStatQuartiles(tmpRes)
        baselineStats[i]=listStats
        g.write("ARI analysis: Mean="+str(listStats[0])+" ,Median="+str(listStats[1])+
            " ,Min="+str(listStats[2])+" ,Max="+str(listStats[3])+" ,Q1="+str(listStats[4])+" ,Q2="+str(listStats[5])+
            " ,Standard Deviation="+str(listStats[6])+ "\n\n")
    f.close()
    g.close()

    experimentalStats={}
    f = open(path+"_ExperimentalResults.txt", "w")
    g = open(path+"_ExperimentalResultsSummary.txt", "w")
    for i in experimentalResults:
        f.write(str(i)+"\n")
        tmpRes=experimentalResults[i]
        f.write(str(tmpRes)+"\n\n")
        g.write(str(i)+"\n")
        listStats=createListStatQuartiles(tmpRes)
        experimentalStats[i]=listStats
        g.write("ARI analysis: Mean="+str(listStats[0])+" ,Median="+str(listStats[1])+
            " ,Min="+str(listStats[2])+" ,Max="+str(listStats[3])+" ,Q1="+str(listStats[4])+" ,Q2="+str(listStats[5])+
            " ,Standard Deviation="+str(listStats[4])+ "\n\n")
    f.close()
    g.close()
    #Write Latex
    writeResultsLatex(baselineStats,path+"_BaselineResLateX.txt")
    writeResultsQuartilesLatex(baselineStats,path+"_BaselineResLateX.txt")
    if(experimentalStats!={}):
        writeResultsLatex(experimentalStats,path+"_ExperimentalResLateX.txt")
        writeResultsQuartilesLatex(experimentalStats,path+"_ExperimentalResLateX.txt")

    return baselineStats,experimentalStats


#allResults= dictionnary
def writeAllResults(allResults,path,options):
    allStats={}
    f = open(path+"_allResults.txt", "w")
    g = open(path+"_resultSummary.txt", "w")
    for i in allResults:
        f.write(str(i)+"\n")
        tmpRes=allResults[i]
        f.write(str(tmpRes)+"\n\n")
        g.write(str(i)+"\n")
        listStats=createListStat(tmpRes)
        allStats[i]=listStats
        g.write("ARI analysis: Mean="+str(listStats[0])+" ,Median="+str(listStats[1])+
            " ,Min="+str(listStats[2])+" ,Max="+str(listStats[3])+" ,Standard Deviation="+str(listStats[4])+ "\n\n")
        #g.write("ARI analysis: Mean="+str(round(np.mean(tmpRes),3))+" ,Median="+str(round(np.median(tmpRes),3))+
        #    " ,Min="+str(min(List_flat))+" ,Max="+str(max(List_flat))+" ,Standard Deviation="+str(round(np.std(List_flat),3))+ "\n\n")
    f.close()
    #Write Latex
    writeResultsLatex(allStats,path+"_resLateX.txt")
    return


#Perform the protocole to lunch CPclust with the given Constraints and Dataset
#We need the IDs of the consensus partition and the constraint set in order to generate the folders and files
def genericGecodeProtocole(idConstrSet,idDataset,idConsensusPartition,currentDataPath):
    generateDirectory(currentDataPath+"/Gecode/output")
    generateDirectory(currentDataPath+"/Gecode/details")
    generateDirectory(currentDataPath+"/Gecode/data")
    resFile=currentDataPath+"/Gecode/output/outputGecode_"+str(DataName[idDataset])+"_CP"+str(idConsensusPartition)+"_CS"+str(idConstrSet)+".txt"
    newDataFile=currentDataPath+'/Gecode/data/gecodeData'+str(DataName[idDataset])+'_CP'+str(idConsensusPartition)+'.txt'
    if os.path.exists(resFile): #to ensure that the data we are going to read is the intended one
        os.remove(resFile)
    #dat=readMatrix(currentDataPath+"/BaselineU/CP0/DistanceMatrix0.txt")
    dat=readMatrix(currentDataPath+"/BaselineU/CP"+str(idConsensusPartition)+"/DistanceMatrix"+str(idConsensusPartition)+".txt")
    distMatToFile(newDataFile,dat) #pas forcement besoin ?

    LauchCPclust(len(GroundTruth[idDataset]),TrueK[idDataset], newDataFile,
                currentDataPath+"/constraints/"+str(DataName[idDataset])+"Constraints"+str(idConstrSet)+".txt",
                currentDataPath+"/Gecode/output/outputGecode_"+str(DataName[idDataset])+"_CP"+str(idConsensusPartition)+"_CS"+str(idConstrSet) )

    if os.path.exists(resFile):
        res=read_result_cpclust(resFile)
        resARI=metrics.adjusted_rand_score(GroundTruth[idDataset],res)
        print("ARI= "+str(resARI))
        minSplit=minimalSplit(dat,res)
        print("MinSplit= "+str(minSplit))
        writeConsensusPartition(currentDataPath+"/Gecode/details/CPdetails_CP"+str(idConsensusPartition)+"_CS"+str(idConstrSet)+".txt",res,minSplit,resARI)
        print(" _________________________________________________________________ \n")
        return resARI
    else:
        print("Error: Gecode result file does not exist")
    return -2

def aggrGecodeProtocole(constraints,idConstrSet,idDataset,idConsensusPartition,currentDataPath):
    resFile=currentDataPath+"/Gecode/Aggr/outputGecode_"+str(DataName[idDataset])+"_CP"+str(idConsensusPartition)+"_CS"+str(idConstrSet)+".txt"
    newDataFile=currentDataPath+'/Gecode/Aggr/gecodeData'+str(DataName[idDataset])+'_CP'+str(idConsensusPartition)+'.txt'
    newCLpath=currentDataPath+'/Gecode/Aggr/newCL'+str(DataName[idDataset])+'_CP'+str(idConsensusPartition)+"_CS"+str(idConstrSet)+'.txt'
    if os.path.exists(resFile): #to ensure that the data we are going to read is the intended one
        os.remove(resFile)
    #dat=readMatrix(currentDataPath+"/BaselineU/CP0/DistanceMatrix0.txt")
    dat=readMatrix(currentDataPath+"/BaselineU/CP"+str(idConsensusPartition)+"/DistanceMatrix"+str(idConsensusPartition)+".txt")
    newDM,newCL,corresp=aggregationML(dat,constraints)
    writeConstraints(newCLpath,newCL)
    #write new contraints in a file, and give it to CPclust
    distMatToFile(newDataFile,newDM) #pas forcement besoin ?

    LauchCPclust(len(GroundTruth[idDataset]),TrueK[idDataset], newDataFile,
                newCLpath,
                currentDataPath+"/Gecode/Aggr/outputGecode_"+str(DataName[idDataset])+"_CP"+str(idConsensusPartition)+"_CS"+str(idConstrSet) )

    if os.path.exists(resFile):
        res=read_result_cpclust(resFile)
        rev=reverseAggregation(corresp,GroundTruth[idDataset],res)
        resARI=metrics.adjusted_rand_score(GroundTruth[idDataset],rev)
        print("ARI= "+str(resARI))
        minSplit=minimalSplit(dat,rev)
        print("MinSplit= "+str(minSplit))
        writeConsensusPartition(currentDataPath+"/Gecode/Aggr/CPdetails_CP"+str(idConsensusPartition)+"_CS"+str(idConstrSet)+".txt",rev,minSplit,resARI)
        print(" _________________________________________________________________ \n")
        return resARI
    else:
        print("Error: Gecode result file does not exist")
    return -2

#Lauch CP clust with the given parameters
#n=number of elements in the dataset
#k=number of clusters in the final partition
#dataFilePath= path to the dataFile
#constrFilePath= path to the file containing the constraints
#outputFilePath= path to the file that will contain the results of the Cpclust execution
def LauchCPclust(n,k,dataFilePath,constrFilePath,outputFilePath):
    #subprocess.check_call(["make"], cwd="/home/mathieu/Documents/Gecode/CP4CC-MG/CP4CC-MG")
    subprocess.run(["./cpclus","-datatype", str(1),"-n", str(n), "-k", str(k), "-obj", str(2), "-searchstrategy", str(1), "-time", str(1000000), "-f",dataFilePath, "-c", constrFilePath, "-op", outputFilePath],  cwd="/home/mathieu/Documents/Gecode/CP4CC-MG/CP4CC-MG")
    #subprocess.run(["./cpclus","-datatype", str(1),"-n", str(n), "-k", str(k), "-obj", str(2), "-searchstrategy", str(1), "-time", str(1000000), "-f",dataFilePath, "-op", outputFilePath],  cwd="/home/mathieu/Documents/Gecode/CP4CC-MG/CP4CC-MG")

#Lauch gecode for diverse datasets. The path need to be change if you want to use it
def GecodeProtocole():
    subprocess.check_call(["make"], cwd="/home/mathieu/Documents/Gecode/CP4CC-MG/CP4CC-MG")
    genericGecodeProtocole(0,0,"/home/mathieu/Documents/Doctorat/Stage/git/stage-involvd-mathieu-guilbert-2021/Devs/protocole/Architecture/halfmoon")
    print(" _________________________________________________________________ ")
    genericGecodeProtocole(0,3,"/home/mathieu/Documents/Doctorat/Stage/git/stage-involvd-mathieu-guilbert-2021/Devs/protocole/Architecture/iris")
    print(" _________________________________________________________________ ")
    genericGecodeProtocole(0,2,"/home/mathieu/Documents/Doctorat/Stage/git/stage-involvd-mathieu-guilbert-2021/Devs/protocole/Architecture/complex9")
    print(" _________________________________________________________________ ")
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run algorithm; need the ID of the dataset, the number of CP to generate and and the number of constraint sets to generate')
    parser.add_argument('dataset', help='int indicating the dataset to test', type=int)
    parser.add_argument('nConsensusPartition', help='number of consensus partition to generate',default=10, type=int)
    parser.add_argument('nConstrSet', help='number of constraint sets to generate',default=5, type=int)
    parser.add_argument('--c', help='specify if the constrained baseline is to be launched', action='store_true')
    parser.add_argument('--ILP25', help='specify if ILP is to be launched with 25 percent of the points of each cluster becoming anchors', action='store_true')
    parser.add_argument('--ILP10', help='specify if ILP is to be launched with 10 percent of the points of each cluster becoming anchors', action='store_true')
    parser.add_argument('--ILP5', help='specify if ILP is to be launched with 5 percent of the points of each cluster becoming anchors', action='store_true')
    parser.add_argument('--aILP', help='lauch alternative ILP (not presented)', action='store_true')
    parser.add_argument('--cILP', help='lauch constrained baseline + ILP (not presented)', action='store_true')
    parser.add_argument('--mpckm', help='specify if MPCKM is to be launched (not presented)', action='store_true')
    parser.add_argument('--cop', help='specify if COP-Kmeans is to be launched', action='store_true')
    parser.add_argument('--copILP', help='specify if ILP is to be lauched on the results of COP-Kmeans', action='store_true')
    parser.add_argument('--copILPa', help='lauch alternative ILP with COP-Kmeans (not presented)', action='store_true')
    parser.add_argument('--ccopILP', help='lauch EAC with COP-Kmeans BPs + ILP (not presented)', action='store_true')
    parser.add_argument('--mzn1', help='specify if the 1st opt criteria coded in Minizinc is to be launched (not presented)', action='store_true')
    parser.add_argument('--mzn2', help='specify if the 2nd opt criteria coded in Minizinc is to be launched (not presented)', action='store_true')
    parser.add_argument('--mzn3', help='specify if the 3rd opt criteria coded in Minizinc is to be launched (not presented)', action='store_true')
    parser.add_argument('--cpclust', help='specify if CPclust is to be lauched (not presented)', action='store_true')
    parser.add_argument('--tripletSimple', help='specify if triplets constraints are to be used in the postprocess', action='store_true')
    parser.add_argument('--tripletInf', help='specify if triplets constraints are to be used in the postprocess (not presented)', action='store_true')
    args = parser.parse_args()

    options=[]
    if(args.c):
        options.append('c')
    if(args.ILP25):
        options.append('ILP25')
    if(args.ILP10):
        options.append('ILP10')
    if(args.ILP5):
        options.append('ILP5')
    if(args.aILP):
        options.append('aILP')
    if(args.cILP):
        options.append('cILP')
    if(args.mpckm):
        options.append('mpckm')
    if(args.cop):
        options.append('cop')
    if(args.copILP):
        options.append('cop')
        options.append('copILP')
    if(args.copILPa):
        options.append('cop')
        options.append('copILPa')
    if(args.mzn1):
        options.append('mzn1')
    if(args.mzn2):
        options.append('mzn2')
    if(args.mzn3):
        options.append('mzn3')
    if(args.cpclust):
        options.append('cpclust')
    if(args.tripletSimple):
        options.append('tripletSimple')
    if(args.tripletInf):
        options.append('tripletInf')

    #DefineMinizincDriver()
    main(args.dataset,args.nConsensusPartition,args.nConstrSet,options)
