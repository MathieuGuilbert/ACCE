
#return the list of sum of distance to the points of l
def listSumDist(l,distMat):
    res=[]
    for n in range(len(distMat)): #pour les n elements
        s=0 #sum
        for e in l:
            s=s+distMat[n][e]
        res.append(s)
    return res

#return the list of the minimum distance to at least one of the points in l
#pas besoin coords du min car on va recalculer pour faire triplet ordre strategie de recherche
def listMinDist(l,distMat):
    res=[]
    #positionMin=[]
    for n in range(len(distMat)): #pour les n elements
        m=1000
        #place=1000
        for e in l:
            if(distMat[n][e]<m):
                m=distMat[n][e]
                #place=e
        res.append(m)
        #positionMin.append(place)
    return res

#update the list of the minimum distance to at least one of the points in l to whom 1 element has been added
def majListMinDist(listmin,l,G,distMat):
    res=[]
    for n in range(len(distMat)): #pour les n elements
        e=l[-1]
        listmin[n]=min(listmin[n],distMat[n][e])
    return listmin

#return the proximity of the clusters relative to e
def proximity(k,element,l,G,distMat):
    distTo=[]
    for i in range(k): #init
        distTo.append(1000)
    for i in range(len(l)):
        c=G[l[i]]
        distTo[c]=min(distTo[c],distMat[element][i])
    return distTo

#proximity: list of distance
#return the order from the closest from the farther cluster relative to a element
def proximityOrder(k,proximity):
    #print("proximity "+str(proximity))
    order=[]
    prec=-1
    while len(order)<k:
        tmpMin=1000
        for i in range(len(proximity)):
            if proximity[i]>=prec and proximity[i]<=tmpMin and (i not in order) :
                tmpMin=proximity[i]
                tmpMinId=i
        order.append(tmpMinId)
        prec=tmpMin
    return order

#return the position of the minimum element of l excluding the indexes in list exc
def excludingMin(exc,l):
    tmpMin=max(l)+1
    ind=0
    for i in range(len(l)):
        if (i not in exc):
            if(l[i]<tmpMin):
                tmpMin=l[i]
                ind=i
    return ind

#ordre distance matrix
def preTreatment1(k,distMat):
    initializers=[]
    #Find the 2 first centroids
    maxDist=-1
    si=1000; sj=1000
    for i in range(len(distMat)):
        for j in range(len(distMat[i])):
            if(maxDist<distMat[i][j]):
                maxDist=distMat[i][j]
                si=i
                sj=j
    initializers.append(si)
    initializers.append(sj)

    #Find the k-2 other centroids
    #For each one we maximize the sum of distances between the new point and the other points already assigned.
    for i in range(k-2):
        listSum=listSumDist(initializers,distMat)
        initializers.append(listSum.index(max(listSum)))

    #main loop
    treated=initializers.copy()
    listMin=listMinDist(treated,distMat)
    while len(treated)!=len(distMat):
        #Take the non-treated point p with the smallest distance to an already treated point d
        listMin=listMinDist(treated,distMat)
        e=excludingMin(treated,listMin)
        #e=listMin.index(min(listMin))
        treated.append(e)

    #print(treated)
    return treated

#change the order of the distance matrix
def reOrderMat(distMat,order):
    newMat=[]
    for e1 in order:
        line=[]
        for e2 in order:
            line.append(distMat[e1][e2])
        newMat.append(line)
    return newMat

#change the order of the distance matrix
def reOrderMat2(distMat,order):
    newMat=[]
    for i in range(len(distMat)): #init
        line=[]
        for j in range(len(distMat)):
            line.append([])
        newMat.append(line)
    for i in range(len(distMat)):
        for j in range(len(distMat)):
            newMat[i][j]=distMat[order[i]][order[j]]
    return newMat

def testTreatment():
    distMat=[[0,2,1,3,1],[2,0,3,1,2],[1,3,0,4,5],[3,1,4,0,3],[1,2,5,3,0]]
    k=3
    ordre=preTreatment1(k,distMat)
    print(ordre)
    nMat=reOrderMat(distMat,ordre)
    print(nMat)

#change the CL constraints according to the new order
def reOrderCL(cl,order):
    newCL=[]
    for p1,p2,t in cl:
        newCL.append((order.index(p1),order.index(p2),t))
        test=order.count(p1)
        if(test>1):
            print("HOLALALALALA "+str(p1))
        test=order.count(p2)
        if(test>1):
            print("HOLALALALALA "+str(p2))
        #newCL.append((order[p1],order[p2],t))
    return newCL

def writeLabels(fileName,labels):
    f = open(fileName, "w")
    f.write(str(labels))
    f.close()
    return

def reversePreTreatment(newOrder,inSP,labels,partition):
    res=[]
    for i in range(len(labels)):
        res.append(partition[newOrder[inSP[i]]])
    print(res)
    return res
