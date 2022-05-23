import numpy as np
import os

#name = a path, for exemple /constraints
#generate a new directory
def generateDirectory(name):
    #path = os.getcwd()
    directoryPath=name
    if not os.path.exists(directoryPath):
        #print(directoryPath)
        os.makedirs(directoryPath) #generate a new directory

#matrix= a matrix (list of list).
#This function is used to write the Co-association matrix and the distance matrix
#write just on one line
def writeMatrix(fileName,matrix):
    f = open(fileName, "w")
    #for m in matrix:
    f.write(str(matrix)+ "\n")
    f.close()
    return

#the matrix needs to be on only one line
def readMatrix(fileName):
    with open(fileName, 'r') as f:
        for line in f:
            line = line.strip()
            if line != '':
                matrix=eval(line)
    return matrix

def writeConsensusPartition(fileName,partition,split,ARI):
    f = open(fileName, "w")
    f.write("ARI: "+str(ARI)+"\n")
    f.write("minSplit: "+str(split)+"\n")
    f.write("Partition: \n")
    f.write(str(partition))
    f.close()
    return

#write the stats of all the consensus partition sets
def writeConsensusPartitionAnalysis(fileName,allMinSplit,consensusARI):
    f = open(fileName, "w")
    f.write("Analysis of all the consensus partition"+"\n\n")
    #ARI
    f.write("Consensus partition ARI analysis: Mean="+str(round(np.mean(consensusARI),3))+" ,Median="+str(round(np.median(consensusARI),3))+
        " ,Min="+str(round(min(consensusARI),3))+" ,Max="+str(round(max(consensusARI),3))+
        " ,Standard Deviation="+str(round(np.std(consensusARI),3))+ "\n")
    #MinSplit
    f.write("Consensus partition MinSplit analysis: ,Mean="+str(round(np.mean(allMinSplit),3))+" ,Median="+str(round(np.median(allMinSplit),3))+
        " ,Min="+str(round(min(allMinSplit),3))+" ,Max="+str(round(max(allMinSplit),3))+
        " ,Standard Deviation="+str(round(np.std(allMinSplit),3))+ "\n")

    #TODO: ajouter %validation contraintes
    f.close()
    return

#CAM: unconstrained co-association matrix
#DistanceMatrix: unconstrained Distance matrix
#constraints=list of tuples with (point 1, point 2, type)
#Create the co-association matrix (CAM) and the distance matrix
def constrainedMatrices(CAM,DistanceMatrix,constraints):
    #initialisation
    newCAM = CAM
    newDistMat=DistanceMatrix
    for (p1,p2,t) in constraints:
        if(t==1): #ML
            newCAM[p1][p2]=1
            newCAM[p2][p1]=1
            newDistMat[p1][p2]=0
            newDistMat[p1][p2]=0
        elif(t==-1): #CL
            newCAM[p1][p2]=0
            newCAM[p2][p1]=0
            newDistMat[p1][p2]=1
            newDistMat[p2][p1]=1
    return newCAM,newDistMat

# 10 dossiers contenant chacun:
#   1 fichier avec la partition consensus sans contraintes son ARI et son MinSplit ,
#   1 fichier contenant sa matrice de co-assotiation et
#   1 fichier pour sa matrice de distance.
#On trouve également à la racine du dossier une analyse des résultats (min,max,mean,median,standard deviation).
def createBaselineFolder(path,allDistMat,allCAM,allCP,allMinSplit,consensusARI):
    writeConsensusPartitionAnalysis(path+"/consensusPartitionAnalysis.txt",allMinSplit,consensusARI)
    for i in range(0,len(allMinSplit)):
        tmpPath=path+"/CP"+str(i)
        generateDirectory(tmpPath)
        writeConsensusPartition(tmpPath+"/ConsensusPartition"+str(i)+".txt",allCP[i],allMinSplit[i],consensusARI[i])
        writeMatrix(tmpPath+"/DistanceMatrix"+str(i)+".txt",allDistMat[i])
        writeMatrix(tmpPath+"/CAM"+str(i)+".txt",allCAM[i])
    return
