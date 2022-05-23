from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
import numpy as np
from scipy.spatial import distance

from preTraitementSpanningTree import *
from vizualization import *

#metrics(label, clustering), metrics(label, partition), get_num_violates(ml, cl,clustering), per_change, model.Runtime, partition
def writeILPRes(DR,path):
    res1,res2,Nviolate,per_change,runtime,partition=DR
    f = open(path, "w")
    f.write("ARI: "+str(res2[0])+"\n")
    f.write("RunTime: "+str(runtime)+"\n")
    #f.write("minSplit: "+str(split)+"\n")
    f.write("Partition: \n")
    f.write(str(partition))
    f.close()
    return partition #=ARI of the partition

#return a distmance matrix between the points cited in l
def computeDistMat(l,data):
    res=[]
    for i in range(len(l)):
        lineI=[]
        for j in range(len(l)):
            lineI.append(distance.euclidean(data[i],data[j]))
        res.append(lineI)
    return res

#return a distmance matrix between the points cited in l
def computeDistMatFromDistMat(l,distMat,ind):
    res=[]
    for i in range(len(l)):
        lineI=[]
        for j in range(len(l)):
            lineI.append(distMat[ind[i]][ind[j]])
        res.append(lineI)
    return res


##--------------------------- Anchors --------------------------

def createAnchorDic(data,distMat,anchors):
    dic={}
    treated=anchors.copy()
    for a in anchors:
        dic[a]=[a]
    for p in range(len(distMat)):
        for a in anchors:
            if (p not in treated) and distMat[p][a]==sorted(distMat[p])[1]:
                dic[a].append(p)
        treated.append(p)

    return dic

#return a list of anchors with only one anchor per cluster in the partition P
def create_K_Anchors(distMat,P,K,data):
    n=len(P)
    listAnchor=[]

    for k in range(K):
        clust=[x for x in range(n) if P[x]==k] #list of the ids of all the datapoints belonging to clust k
        #matDistClust=computeDistMat(clust,data) #convert into euclidean matrix

        if len(clust)>0:
            minSum=100000
            kAnchor=-1
            for j in clust:
                #tempSum=sum([matDistClust[j][x] for x in i_elem])
                tempSum=sum([distMat[j][x] for x in clust])
                if tempSum<minSum:
                    minSum=tempSum
                    kAnchor=j
            listAnchor.append(kAnchor)

    return listAnchor

#return a list of anchors with only one anchor per cluster in the partition P
def create_Anchor_clust(distMat,clust):
    kAnchor=-1
    if len(clust)>0:
        minSum=100000
        for j in clust:
            tempSum=sum([distMat[j][x] for x in clust])
            if tempSum<minSum:
                minSum=tempSum
                kAnchor=j
    return kAnchor


#return a list of anchors with only one anchor per cluster in the partition P
def create_Anchor_clust_cond(distMat,clust,threshold):
    kAnchor=-1
    sat=False
    sumSat=0
    allSum=0
    if len(clust)>0:
        minSum=100000
        for j in clust:
            #tempSum=sum([distMat[j][x] for x in clust])
            tempSum=0
            tempList=[]
            for x in clust:
                tempSum+=distMat[j][x]
                tempList.append(distMat[j][x])
            if tempSum<minSum:
                minSum=tempSum
                minlist=tempList
                kAnchor=j
            if (threshold >= tempSum/len(clust) ):
                sumSat+=1
            allSum+=tempSum
        meanDist=minSum/len(clust)
        print("NB sat: "+str(sumSat)+" , allSum: "+str(allSum)+" , meanAllSum: "+str(allSum/(len(clust)*len(clust)) ))
        sat= (threshold >= meanDist )
    return kAnchor,sat,meanDist,minlist


#-------Complex anchor computations -----------

#return a list of anchors (id of the anchors in the dataset) with a maximum of h elements
def createAnchorsAdaptative(P,K,data):
    n=len(P)
    listAnchor=[]

    for k in range(K):
        sat=False
        clust=[x for x in range(n) if P[x]==k] #list of the ids of all the datapoints belonging to clust k

        matDistClust=computeDistMat(clust,data) #convert into euclidean matrix
        Y = csr_matrix(matDistClust)
        Tcsr = minimum_spanning_tree(Y)

        h=0
        threshold=np.percentile(matDistClust,25) #a bit lower than first quartile

        while(not sat):
            h=h+1
            #print("h="+str(h))
            cutTreeH=cutSpanningTree(h,Tcsr)
            H_n_comp, Ph = connected_components(csgraph=cutTreeH, directed=False, return_labels=True)
            allDist=[]
            kAnchors=[]

            #Then we can define the anchors
            for i in range(0,h):
                #print("defining anchors "+str(i))
                i_elem=[p for p in range(len(Ph)) if Ph[p] == i] #put the position of all the occurences of i in the partition
                if len(i_elem)>0:
                    minSum=10000
                    minSumPos=-1
                    for j in i_elem:
                        tempSum=sum([matDistClust[j][x] for x in i_elem])
                        if tempSum<minSum:
                            minSum=tempSum
                            minSumPos=j
                    #print(minSumPos)
                    anchor=minSumPos
                    #print(str(minSum)+" < "+str(threshold))
                    allDist.append(minSum)
                    kAnchors.append(clust[anchor])

            #print(allDist)
            #print(np.mean(allDist))
            sat=(np.mean(allDist)<=threshold)

        print(str(h)+" / "+str(len(clust)))
        listAnchor=listAnchor+kAnchors

    print("length listAnchor "+str(len(listAnchor)))
    return listAnchor

#---

#return a list of anchors (id of the anchors in the dataset) with a maximum of h elements
#At each step, if the condition is not satsfied, it devides the subcluster into 2
def createAnchorsDynamic(K, data, P, per, distMat):
    #print(P)
    n=len(P)
    listAnchor=[]
    for k in range(K):
        #print(" cluster "+str(k))
        clustK=[x for x in range(n) if P[x]==k] #list of the ids of all the datapoints belonging to clust k
        newInd=[j for j in range(n) if P[j]==k]

        distMatClust=computeDistMat(clustK,data) #convert into euclidean matrix
        Y = csr_matrix(distMatClust)
        MST = minimum_spanning_tree(Y)
        kAnchors=[]
        #threshold=np.percentile(distMatClust,per)

        #anch,sat,meanDist,minlist=create_Anchor_clust_cond(distMat,clustK,threshold)
        #if(sat):
        #    kAnchors.append(anch)
        #    print("Only 1 anchor for clust "+str(k)+" with len "+str(len(clustK))+" , Mean: "+str(np.mean(distMatClust))+" , Median: "+str(np.median(distMatClust))+" , threshold: "+str(threshold)+" , meanDist: "+str(meanDist)+" ,list: "+str(minlist) )
        #else:
        recKanchor(data,clustK, distMat, MST, per, newInd ,kAnchors)
        #print("Mean: "+str(np.mean(distMatClust))+" , Median: "+str(np.median(distMatClust))+" , threshold: "+str(threshold)+" , meanDist: "+str(meanDist)+" , nb Anchor: "+str(len(kAnchors)))
            #recKanchor(clustK, distMatClust, MST, threshold, clustK ,kAnchors)

        listAnchor=listAnchor+kAnchors

    print("nb of Dynamic anchors: "+str(len(listAnchor)))
    return listAnchor

#recursive process
#ind=indices
def recKanchor(data,clustK, distMatClust, MST, per, ind, listAnchor):
    #print("Old MST: "+str(MST))
    #print("old ind "+str(ind))
    cutTree=cutSpanningTree(2,MST)
    H_n_comp, Ph = connected_components(csgraph=cutTree, directed=False, return_labels=True)
    #print("Ph : "+str(Ph))
    #PROBLEM: peut pas obtenir direct les comp connexe -> on doit recalculer le MST ou faire suppr à la main

    allClust=[[x for x in range(len(Ph)) if Ph[x]==0],[x for x in range(len(Ph)) if Ph[x]==1]]
    for i in range(H_n_comp):
        clust=allClust[i] #[x for x in range(len(Ph)) if Ph[x]==i]
        #distMatTMP=computeDistMat([ind[x] for x in clust],data)
        #distMatTMP=computeDistMat(clust,data)

        #clustData=data.copy()
        #sorted_indecies_to_delete = sorted([x for x in range(n) if P[x]!=k], reverse=True)
        #for index in sorted_indecies_to_delete:
        #    del clustData[index]
        #distMatTMP=computeDistMat(clust,clustData)

        distMatTMP=computeDistMatFromDistMat(clust,distMatClust,ind)
        flat_list = [item for sublist in distMatTMP for item in sublist if (item!=0)]
        if(flat_list==[]):
            flat_list.append(0.0)
        #threshold=np.percentile(distMatTMP,per)
        threshold=np.percentile(flat_list,per)
        #threshold=(per*max(flat_list))/100

        newInd=[] #newInd=[ind[j] for j in clust]
        for j in clust:
            newInd.append(ind[j])

        #print("new defined ind "+str(newInd)+" ; old ind "+str(ind))
        #print("i "+str(i)+" len clust "+str(len(clust))+" ; str clust: "+str(clust))
        if i==0:
            cc=extractComp(MST,allClust[1])
            print(clust,allClust[1],cc)
        else:
            cc=extractComp(MST,allClust[0])

        #Then we can define the anchor
        if len(clust)>1:
            minSum=10000
            minSumPos=-1
            for j in clust:
                #tempSum=sum([distMatClust[j][x] for x in clust])
                #print("j: "+str(j)+" "+str(ind[j]))
                #print("x: "+str(len(clust)-1)+" ; len ind "+str(len(ind)))
                #print("Len dist Mat "+str(len(distMatClust)),"Len dist mat[0] "+str(len(distMatClust[0])))
                #distMatClust[ind[j]][ind[len(clust)-1]]

                #tempSum=sum([distMatClust[ind[j]][ind[x]] for x in clust])
                tempSum=0
                for x in clust:
                    di=distMatClust[ind[j]][ind[x]]
                    if(di!=0.0):
                        tempSum+=di
                        #print(di,j,x,ind[j],ind[x])

                if tempSum<minSum:
                    minSum=tempSum
                    minSumPos=j
            anchor=minSumPos
        else:
            #print("len clust = 1")
            anchor=clust[0]
            minSum=0

        #meanDist=minSum/len(clust) #FAUX ?
        if(len(clust)>1):
            meanDist=minSum/(len(clust)-1)
        else:
            meanDist=minSum/(len(clust))

        ##print("In rec "+str(i)+" with "+str(len(clust))+" points :  Mean: "+str(np.mean(distMatTMP))+" , Median: "+str(np.median(distMatTMP))+" , threshold: "+str(threshold)+" ,MinSum: "+str(minSum)+" , MeanDist: "+str(meanDist)+" , DistMat: "+str(flat_list))

        if(meanDist <= threshold):
            listAnchor.append(ind[anchor])
            #listAnchor.append(anchor)
        else:
            recKanchor(data, clust, distMatClust, cc, per, newInd, listAnchor)
    return

def extractComp(MST,toDelete):
    #print("ToDelete "+str(toDelete))
    mstArray=MST.toarray()
    del0=np.delete(mstArray, toDelete, 0)
    del1=np.delete(del0, toDelete, 1)
    print(len(del1))

    cc=csr_matrix(del1)
    #print("end cc "+" : "+str(cc))
    return cc

#-----------------------AllocationMatrix----------------------------

#Local matAlloc (our main approach)
def createAltMatAlloc(data,K,d,path,Pk,method,distMat):
    n=len(Pk) #number of elements in the dataset

    #listAnchor=createAnchorsAdaptative(Pk,K,data)
    listAnchor=createAnchorsDynamic(K, data, Pk,30,distMat)
    #print("Number of alt anchors: "+str(len(listAnchor)))

    #listAnchor=create_K_Anchors(distMat,Pk,K,data)

    anchorPos=[]
    for a in listAnchor:
        anchorPos.append(data[a])

    #Computation of the Matrix
    M= [ [ -1 for i in range(K) ] for j in range(n) ]
    for i in range(0,n):
        for k in range(0,K):
            isAnch=False
            for anch in listAnchor:
                if(Pk[anch]==k):
                    isAnch=True
                    if(M[i][k]==-1):
                        M[i][k]=distance.euclidean(data[anch],data[i]) #Mik is the euclidian distance between the point i and the closest anchor belonging to the cluster k
                    else:
                        M[i][k]=min(M[i][k],distance.euclidean(data[anch],data[i]))

    #Normalization
    Mnorm=[ M[i].copy() for i in range(n) ]
    for i in range(0,n):
        sumLine=sum(Mnorm[i])
        for k in range(0,K):
            Mnorm[i][k]=(sumLine-Mnorm[i][k])
        sumLine2=sum(Mnorm[i])
        for k in range(0,K):
            Mnorm[i][k]=(Mnorm[i][k]/sumLine2)

    #Save the intermediary results
    #writeAnchors(data,H,K,Ph,Ph_updated,listAnchor,anchorPos,Pk,d,path)
    vizualizeAnchorsOnPartition(K,data,Pk,listAnchor,path+"/anchorsDyn"+method+"_"+str(d)+".png")
    vizualizeNeutralAnchors(anchorPos,K,data,path+"/anchorsDynNeutral"+method+"_"+str(d)+".png")
    return Mnorm,listAnchor
