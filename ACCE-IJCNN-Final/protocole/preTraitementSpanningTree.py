from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
import numpy as np

k=2
distMat=[[0, 2, 1.1, 3, 1],
        [2, 0, 3, 1, 2],
        [1.1, 3, 0, 4, 5],
        [3, 1, 4, 0, 3],
        [1, 2, 5, 3, 0]]

X = csr_matrix([[0, 8, 0, 3],
                [0, 0, 2, 5],
                [0, 0, 0, 6],
                [0, 0, 0, 0]])
Y = csr_matrix(distMat)

#return a list of list where i-th list contains the points belonging to the i-th label
def defclust(k,labels):
    res=[]
    for clust in range(k): #init
        res.append([])

    for clust in range(k):
        for i in range(len(labels)):
            if(labels[i]==clust):
                res[clust].append(i)
    return res


def medianInList(l,distMat):
    tmp=10000
    ide=0
    for i in range(len(l)):
        s=0
        for j in range(len(l)):
            s+=distMat[l[i]][l[j]]
        #print(str(l[i])+" "+str(s))
        if(s<tmp):
            tmp=s
            ide=l[i]
    return ide

#cl= list of tuple in format (point1,point2,type)
def newOrder(k,seeds,cl,lab):
    byClustCL=[] #advancedCLorder
    byClustU=[]
    inCL=[] #list of elements existing in a CL
    for i in range(k): #init
        byClustCL.append([])
        byClustU.append([])
    for (p1,p2,t) in cl:
        if((p1 not in inCL) and (p1 not in seeds)):
            #print("test1")
            byClustCL[lab[p1]].append(p1)
            inCL.append(p1)
        if((p2 not in inCL) and (p2 not in seeds)):
            #print("test2")
            byClustCL[lab[p2]].append(p2)
            inCL.append(p2)
    clOrder = [item for sublist in byClustCL for item in sublist] #flatten
    #print(byClustCL)
    #print(clOrder)

    for i in range(len(lab)):
        if ((i not in seeds) and (i not in inCL)):
            byClustU[lab[i]].append(i)

    unconstrOrder=[item for sublist in byClustU for item in sublist]

    res=seeds.copy()
    res+=clOrder+unconstrOrder
    #print("Previous order2: "+str(res))
    #return res
    #ICI: test revOrder
    RevOrder=[]
    for i in range(len(res)):
        RevOrder.append([])
    for i in range(len(res)):
        RevOrder[res[i]]=i
    #print(" RevOrder: "+str(RevOrder))
    return res,RevOrder



#sp= a spanning tree
#return the the same spanning tree but with the k-1 heaviest edges
def cutSpanningTree(k,sp):
    res=sp.toarray().copy()
    for h in range(k-1):
        maxtemp=0
        a,b=(0,0)
        for i in range(len(res)):
            for j in range(len(res)):
                if res[i][j]>maxtemp:
                    maxtemp=res[i][j]
                    a,b=(i,j)
        res[a][b]=0
    return res

def preTreatment2(k,distMat,cl):
    Y = csr_matrix(distMat)
    Tcsr = minimum_spanning_tree(Y)
    cSP=cutSpanningTree(k,Tcsr)
    n_components, labels = connected_components(csgraph=cSP, directed=False, return_labels=True)
    G=defclust(k,labels)
    seeds=[]
    for i in range(k):
        seeds.append(medianInList(G[i],distMat))
    order,Rorder=newOrder(k,seeds,cl,labels)
    return order,Rorder

def oldTest():
    Tcsr = minimum_spanning_tree(Y)
    print(Tcsr)
    print(Tcsr.toarray())
    testComp=cutSpanningTree(k,Tcsr)
    n_components, labels = connected_components(csgraph=testComp, directed=False, return_labels=True)
    print()
    print(n_components)
    print(labels)
    G=defclust(k,labels)
    print(G)
    seeds=[]
    for i in range(k):
        seeds.append(medianInList(G[i],distMat))
    print(seeds)
    print(newOrder(k,seeds,[(0,4,-1)],labels))
    return

#res=preTreatment2(k,distMat,[])
#print(res)
