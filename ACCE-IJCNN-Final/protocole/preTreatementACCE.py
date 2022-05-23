from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from sklearn.cluster import AgglomerativeClustering
import numpy as np
#from math import dist
from scipy.spatial import distance

from preTraitementSpanningTree import *
from vizualization import *
from preTreatmentAlternativeILP import *

#write a distance matrix in a txt file with the standard needed by CPclust
#name is the intended name of the txt file
#data is a list of list
def distMatToFile(name,data):
    f = open(name, "w")
    firstline=""
    for i in range(len(data[0])):
            #value=str(data[0][i])
            value=str(int(float(data[0][i])*10000))
            firstline+=value
            if(not (i==len(data[0])-1 ) ):
                firstline+=" "
    f.write(firstline)
    #for all the other lines
    for j in range(1,len(data)):
        line="\n"
        for i in range(len(data[0])):
            #value=str(data[j][i])
            value=str(int(float(data[j][i])*10000))
            line+=value
            if(not (i==len(data[0])-1 ) ):
                line+=" "
        f.write(line)
    f.close()
    return

#read a conform datafile using floats
def read_data_cpclust(datafile):
    data = []
    with open(datafile, 'r') as f:
        for line in f:
            line = line.strip()
            if line != '':
                tmp=line.split()
                for i in range(len(tmp)):
                    tmp[i]=(float(tmp[i]))
                data.append(tmp)
    #print(len(data))
    return data


#read a resulting file from CPclust and convert into an array
def read_result_cpclust(resultfile):
    res = []
    with open(resultfile, 'r') as f:
        for line in f:
            line = line.strip()
            if line != '':
                tmp=line.split()
                clust=""
                for i in range(len(tmp)):
                    clust=clust+str(tmp[i])
                res.append(int(clust))
                #res.append(int(line[0]))
    #print(res)
    return res

#-----------------------------------------------------------------------------------------------------------------------

#return a list of anchors (id of the anchors in the dataset) with a maximum of h elements
#(our main approach)
def createAnchorsByClustEucl(P,K,data,div,d,path,method):
    n=len(P)
    listAnchor=[]

    for k in range(K):
        kAnchors=[]
        scaledAnchors=[]
        clust=[x for x in range(n) if P[x]==k] #list of the ids of all the datapoints belonging to clust k
        if(len(clust)>=div):
            h=int(len(clust)/div)
        else:
            h=1

        #matDistClust=computeDistMat(clust,data) #convert into euclidean matrix
        #Y = csr_matrix(matDistClust)
        #Tcsr = minimum_spanning_tree(Y)
        #cutTreeH=cutSpanningTree(h,Tcsr)
        #H_n_comp, Ph = connected_components(csgraph=cutTreeH, directed=False, return_labels=True)

        #using Single Link
        clustData=data.copy()
        sorted_indecies_to_delete = sorted([x for x in range(n) if P[x]!=k], reverse=True)
        for index in sorted_indecies_to_delete:
            del clustData[index]

        matDistClust=computeDistMat(clust,clustData)

        if(len(clust)>1):
            mat = np.array(matDistClust)
            clustering = AgglomerativeClustering(affinity='precomputed',linkage='single',n_clusters=h).fit(matDistClust)
            #clustering = AgglomerativeClustering(affinity='precomputed',linkage='complete',n_clusters=h).fit(mat)
            clustering
            Ph=clustering.labels_
        else:
            Ph=[0]

        #Then we can define the anchors
        for i in range(0,h):
            i_elem=[p for p in range(len(Ph)) if Ph[p] == i] #put the position of all the occurences of i in the partition
            if len(i_elem)>0:
                minSum=10000
                minSumPos=-1
                for j in i_elem:
                    tempSum=sum([matDistClust[j][x] for x in i_elem])
                    if tempSum<minSum:
                        minSum=tempSum
                        minSumPos=j
                anchor=minSumPos
                kAnchors.append(clust[anchor])
                scaledAnchors.append(anchor)

        listAnchor=listAnchor+kAnchors

    return listAnchor

#----- Local matAlloc (our main approach) ----------
def creatEuclMatAlloc_AbC(data,K,d,path,Pk,div,method):
    n=len(Pk) #number of elements in the dataset

    listAnchor=createAnchorsByClustEucl(Pk,K,data,div,d,path,method) #anchors are ids here
    print("Number of AbC anchors: "+str(len(listAnchor)))
    #listAnchor=createAnchorsAdaptative(Pk,K,data)

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

    Mnorm=originalNormalization(M,K,n)

    #Save the intermediary results
    #writeAnchors(data,H,K,Ph,Ph_updated,listAnchor,anchorPos,Pk,d,path)
    vizualizeAnchorsOnPartition(K,data,Pk,listAnchor,path+"/anchOnPartitionAbC_"+method+"_"+str(d)+"_div"+str(div)+".png")
    vizualizeNeutralAnchors(anchorPos,K,data,path+"/anchNeutralAbC_"+method+"_"+str(d)+"_div"+str(div)+".png")
    print(anchorPos[1200])
    return Mnorm

#normalization as in the final paper
#D=matrix where Dik contains the distance from point i to the closest anchor in cluster k
#K=number of clusters
#n=number of elements
def simplerNormalization(D,K,n):
    M=[ D[i].copy() for i in range(n) ]
    for i in range(0,n):
        sumLine=sum(D[i])
        for k in range(0,K):
            M[i][k]=(1-D[i][k]/sumLine)/(K-1)

    return M

#normalization used in the first submitted version of the paper
def originalNormalization(D,K,n):
    Mnorm=[ D[i].copy() for i in range(n) ]
    for i in range(0,n):
        sumLine=sum(Mnorm[i])
        for k in range(0,K):
            Mnorm[i][k]=(sumLine-Mnorm[i][k])
        sumLine2=sum(Mnorm[i])
        for k in range(0,K):
            Mnorm[i][k]=(Mnorm[i][k]/sumLine2)
    return Mnorm



#write in a file all the computed partitions and values during the creation of the anchors
def writeAnchors(data,H,K,Ph,Ph_updated,listAnchor,anchorPos,Pk,d,path):
    f = open(path+"/Anchors_"+str(d)+".txt", "w")
    f.write("-- Anchors details --\n\n")
    f.write("H= "+str(H)+"\n")
    f.write("final number of anchors: "+str(len(anchorPos))+"\n\n")
    f.write("Default partition with H clusters: \n")
    f.write(str(Ph)+"\n")
    f.write("Updated partition (where clusters with only one element have been merged with their closest neighbor): \n")
    f.write(str(Ph_updated)+"\n")
    f.write("Anchors ID: \n")
    f.write(str(listAnchor)+"\n")
    f.write("Anchors position: \n")
    f.write(str(anchorPos)+"\n\n")
    f.write("Partition with K clusters: \n")
    f.write(str(Pk)+"\n")
    return


def writeMZNcrit2(path,constraints):
    f = open(path+".mzn", "w")
    f.write("include \"globals.mzn\"; \n")
    f.write("\n")
    f.write("int: n; % number of points \n")
    f.write("int: k_min; \n")
    f.write("int: k_max; \n")
    f.write("\n")
    #f.write("array[1..n, 1..k_max] of float: M; \n")
    f.write("array[1..n, 1..k_max] of int: M; \n")
    f.write("array[1..n] of int: P; \n")
    f.write("array[1..n] of var 1..k_max: G; \n")
    f.write("\n")
    #f.write("var min(i,j in 1..n where i<j)(dist[i,j]) .. max(i,j in 1..n where i<j)(dist[i,j]) : S; \n")
    #OBJ: #minimize $max(M_{ik}(G[i]\neq G'[i]\,\&\, G[i]=k))$
    #f.write("var min(i in 1..n and k in 1..k where P[i]!=G[i] and P[i]=k)(M[i,j]) .. max(i in 1..n and k in 1..k where P[i]!=G[i] and P[i]=k)(M[i,j]) : O; \n")
    #f.write("var min(i in 1..n, k in 1..k_max where P[i]!=G[i] /\ P[i]=k)(M[i,k]) .. max(i in 1..n, k in 1..k_max where P[i]!=G[i] /\ P[i]=k)(M[i,k]) : Obj; \n") DID NOT WORK yet, MiniZinc: type error: type error in operator application for `'..''. No matching operator found with left-hand side type `var float' and right-hand side type `var float. from what i understand, MZN does not like that I use a variable to define another variable -> Stack Overflow ?
    #f.write("var min(i in 1..n, k in 1..k_max)(M[i,k]) .. max(i in 1..n, k in 1..k_max)(M[i,k]) : Obj; \n")
    f.write("var min(M) .. max(M) : Obj; \n")
    f.write("\n")
    #Optimization criterion: #f.write("constraint forall (i,j in 1..n where i<j) ( dist[i,j] < S -> G[i] == G[j] ); \n")
    f.write("constraint forall (i in 1..n, k in 1..k_max where P[i]=k) ( M[i,k] > Obj -> P[i]=G[i]); \n\n")
    #f.write("constraint forall(k in 1..k_max)(sum(i in 1..n)(G[i]==k)>0);") #We need to impose that there is k clusters

    #Trying to reduce the area of research
    #f.write("array[1..n] of var -1..max(i in 1..n, k in 1..k_max)(M[i,k]): is_moved; \n")
    #f.write("constraint forall (i in 1..n)(P[i]=G[i] -> is_moved[i]=-1); \n")
    #f.write("constraint forall (i in 1..n, k in 1..k_max where P[i]=k)(P[i]!=G[i] -> is_moved[i]=M[i,k]); \n")
    #f.write("predicate atleast(int: a, array[int] of var int: is_moved, int: Obj) ; \n")

    f.write("\n")
    for (p1,p2,t) in constraints:
        if t==1: #1 -> ML
            f.write("constraint G["+str(p1+1)+"]=G["+str(p2+1)+"]; \n")
        if t==-1: #-1 -> CL
            f.write("constraint G["+str(p1+1)+"]!=G["+str(p2+1)+"]; \n")
    #constraints
    f.write("%%%%%%%%%%%%% \n")
    f.write("\n")
    f.write("solve ::int_search(G, first_fail, indomain_min, complete) minimize Obj; \n")
    f.write("output [\"G = \(G)\\nObj=\(Obj)\"]; \n") #f.write("output [\"G = \(G)\\nObj=\(Obj)\\nis_moved=\(is_moved)\"]; \n")
    return

def writeMZNcrit3(path,constraints):
    f = open(path+".mzn", "w")
    f.write("include \"globals.mzn\"; \n")
    f.write("\n")
    f.write("int: n; % number of points \n")
    f.write("int: k_min; \n")
    f.write("int: k_max; \n")
    f.write("\n")
    f.write("array[1..n, 1..k_max] of int: M; \n")
    f.write("array[1..n] of int: P; \n")
    f.write("array[1..n] of var 1..k_max: G; \n")
    f.write("\n")

    f.write("var min(M) .. max(M) : Obj; \n")
    f.write("\n")
    #Optimization criterion:
    f.write("constraint forall (i in 1..n, k in 1..k_max where G[i]=k) ( M[i,k] < Obj -> P[i]=G[i]); \n")
    #f.write("constraint forall(k in 1..k_max)(sum(i in 1..n)(G[i]==k)>0);") #We need to impose that there is k clusters

    f.write("\n")
    for (p1,p2,t) in constraints:
        if t==1: #1 -> ML
            f.write("constraint G["+str(p1+1)+"]=G["+str(p2+1)+"]; \n")
        if t==-1: #-1 -> CL
            f.write("constraint G["+str(p1+1)+"]!=G["+str(p2+1)+"]; \n")

    f.write("%%%%%%%%%%%%% \n")
    f.write("\n")
    f.write("solve ::int_search(G, first_fail, indomain_min, complete) maximize Obj; \n")
    f.write("output [\"G = \(G)\\nObj=\(Obj)\"]; \n")
    return

def writeMZNcrit1(path,constraints):
    f = open(path+".mzn", "w")
    f.write("include \"globals.mzn\"; \n")
    f.write("\n")
    f.write("int: n; % number of points \n")
    f.write("int: k_min; \n")
    f.write("int: k_max; \n")
    f.write("\n")
    f.write("array[1..n, 1..k_max] of int: M; \n")
    f.write("array[1..n] of int: P; \n")
    f.write("array[1..n] of var 1..k_max: G; \n")
    f.write("\n")

    f.write("var n*min(M)..n*max(M): S; \n") #pourrait restreindre plus le domaine. Pour l'instant n car normalisé.
    f.write("\n")
    #Optimization criterion:
    f.write("constraint S=sum(i in 1..n, k in 1..k_max where G[i]=k)(M[i,k]); \n")

    f.write("\n")
    for (p1,p2,t) in constraints:
        if t==1: #1 -> ML
            f.write("constraint G["+str(p1+1)+"]=G["+str(p2+1)+"]; \n")
        if t==-1: #-1 -> CL
            f.write("constraint G["+str(p1+1)+"]!=G["+str(p2+1)+"]; \n")

    f.write("%%%%%%%%%%%%% \n")
    f.write("\n")
    f.write("solve ::int_search(G, first_fail, indomain_min, complete) maximize S; \n")
    f.write("output [\"G = \(G)\\nObj=\(S)\"]; \n")
    return

#convert a distance matrix into a dzn file
#name is the intended name of the dzn file
#data is a list of list
#n is the number of elements
#k_min and k_max are the limits of the number of clusters
#MatAlloc an allocation matrix
#P a partition
def writeDZN_MatAlloc(name,data,n,k_min,k_max,MatAlloc,P):
    #write DZN
    f = open(name+".dzn", "w")
    f.write("n="+str(n)+";")
    f.write("\nk_min="+str(k_min)+";")
    f.write("\nk_max="+str(k_max)+";")
    #AllocationMatrix
    #for the first line
    firstline="\nM=[| "
    for i in range(len(MatAlloc[0])):
            #value=str(round(MatAlloc[0][i],3))
            value=str(int(MatAlloc[0][i]*1000))
            firstline+=value
            if(not (i==len(MatAlloc[0])-1 ) ):
                firstline+=", "
    f.write(firstline)
    #for all the other lines
    for j in range(1,len(MatAlloc)):
        line="\n | "
        for i in range(len(MatAlloc[0])):
            #value=str(round(MatAlloc[j][i],3))
            value=str(int(MatAlloc[j][i]*1000))
            line+=value
            if(not (i==len(MatAlloc[0])-1 ) ):
                line+=", "
        f.write(line)
    f.write(" |];")

    #old partition P
    part="\n P=["
    for i in range(len(P)):
        value=str(P[i]+1)
        part+=value
        if(not (i==len(P)-1 ) ):
            part+=", "
    part+=" ];"
    f.write(part)

    f.close()
    return

def accordanceMoveConstr(P,G,constraints):
    print(P)
    print(G)
    nbr=0
    PdifG=0
    for i in range(len(P)):
        if( int(P[i])!=(int(G[i])-1) ):
            PdifG+=1
            for e1,e2,t in constraints:
                if(e1==i or e2==i):
                    nbr+=1
    return PdifG,nbr
