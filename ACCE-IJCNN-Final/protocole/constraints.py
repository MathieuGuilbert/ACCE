import random

#Nr = number of constraint set to generate
#Nc = number of constraints to generate
#data = dataset
#labels = GroundTruth labels of the data
#name: name (including path) of the files that will be created
#Attention: Warning ! It is possible to only obtain ML or CL constraints
def createConstraints(Nr,Nc,data,labels,name):
    constrSets=[]
    for a in range(0,Nr):
        ml=[]
        cl=[]
        cons=[]
        for i in range(0,Nc):
            r = random.randint(0,len(data)-1)
            r1 = r
            #To avoid that the 2 points are the same
            while( r1==r or ((r,r1)in ml) or ((r1,r)in ml) or ((r,r1)in cl) or ((r1,r)in cl) ):
                r1 = random.randint(0,len(data)-1)

            if(labels[r]==labels[r1]):
                ml.append((r,r1))
                cons.append((r,r1,1))
            else:
                cl.append((r,r1))
                cons.append((r,r1,-1))
        constrSets.append(cons)
        filename=name+str(a)+".txt"
        writeConstraints(filename,cons)
    return constrSets

#create only ML constraints
def createMLs(Nr,Nc,data,labels,name):
    constrSets=[]
    for a in range(0,Nr):
        ml=[]
        cons=[]
        for i in range(0,Nc):
            r = random.randint(0,len(data)-1)
            r1 = r
            #Pour eviter que les deux points soient les même
            while( r1==r or ((r,r1)in ml) or ((r1,r)in ml) or (labels[r]!=labels[r1])):
                r1 = random.randint(0,len(data)-1)
            if(labels[r]==labels[r1]):
                ml.append((r,r1))
                cons.append((r,r1,1))
        constrSets.append(cons)
        filename=name+str(a)+".txt"
        writeConstraints(filename,cons)
    return constrSets

def createCLs(Nr,Nc,data,labels,name):
    constrSets=[]
    for a in range(0,Nr):
        ml=[]
        cons=[]
        for i in range(0,Nc):
            r = random.randint(0,len(data)-1)
            r1 = r
            #Pour eviter que les deux points soient les même
            while( r1==r or ((r,r1)in ml) or ((r1,r)in ml) or (labels[r]==labels[r1])):
                r1 = random.randint(0,len(data)-1)
            if(labels[r]!=labels[r1]):
                ml.append((r,r1))
                cons.append((r,r1,-1))
        constrSets.append(cons)
        filename=name+str(a)+".txt"
        writeConstraints(filename,cons)
    return constrSets

#return a list with the CLs and a list with the MLs
def extractCLandML3D(constraints):
    CL=[]
    ML=[]
    for e1,e2,t in constraints:
        if t==-1:
            CL.append((e1,e2,t))
        else:
            ML.append((e1,e2,t))
    return CL,ML

#return a list with the CLs and a list with the MLs
def extractCLandML2D(constraints):
    CL=[]
    ML=[]
    for e1,e2,t in constraints:
        if t==-1:
            CL.append((e1,e2))
        else:
            ML.append((e1,e2))
    return CL,ML

#cons= a constraint set
def writeConstraints(fileName,cons):
    f = open(fileName, "w")
    for (a,b,t) in cons:
        f.write(str(a)+" "+str(b)+" "+str(t)+ "\n")
    f.close()
    return

#read a file of constraints and returns a list of triplets
def read_constraints(consfile):
    c = []
    with open(consfile, 'r') as f:
        for line in f:
            line = line.strip()
            if line != '':
                line = line.split()
                constraint = (int(line[0]),int(line[1]),int(line[2]))
                c.append(constraint)
    return c

#test if a CL is in contradiction with the labels
def testValidityCL(cl,labels):
    for (p1,p2,t) in cl:
        if t==-1:
            if labels[p1]==labels[p2]:
                print("Erreur CL validity")
                print(str(p1)+" "+str(p2))
                return False
    return True

#test if a ML is in contradiction with the labels
def testValidityML(ml,labels):
    for (p1,p2,t) in ml:
        if t==1:
            if labels[p1]!=labels[p2]:
                print("Erreur ML validity")
                print(str(p1)+" "+str(p2))
                return False
    return True

#returns a percentage corrresponding to the number of constraints verified in the given partition
def verify_constraints(partition,constraints):
    nVerif=0; nConstr=len(constraints)
    for e1,e2,t in constraints:
        if t==-1:
            if partition[e1]!=partition[e2]:
                nVerif+=1
        else: #if t==1
            if partition[e1]==partition[e2]:
                nVerif+=1
    per=(nVerif*100)/nConstr
    return per

def verify_multiple_constrSets(partition,constrSets):
    resVer=[]
    for constraints in constrSets:
        resVer.append(verify_constraints(partition,constraints))
    return resVer

def verify_corresponding_constr(partSets,constrSets):
    resVer=[]
    for constraints in constrSets:
        resVer.append(verify_constraints(partition,constraints))
    return resVer

# ------------ Prairwise prop

#constraint: a Ml constraint
#anchors: list of anchors
#distMat: euclidean distance matrix
def MLpropagation(constraint,anchors,anchorDic,distMat):
    newConstr=[]
    (a,b,t)=constraint
    anchorA=anchors[0]

    minDist=10000
    for anch in anchors:
        if(a in anchorDic[anch]):
            anchorA=anch

    for p in anchorDic[anchorA]:
        if distMat[p][b] <= distMat[a][b]:
            newConstr.append((p,b,t))

    anchorB=anchors[0]

    minDist=10000
    for anch in anchors:
        if(b in anchorDic[anch]):
            anchorB=anch

    for p in anchorDic[anchorB]:
        if distMat[a][p] <= distMat[a][b]:
            newConstr.append((a,p,t))

    return newConstr

#propagate CL constraints
def CLpropagation(constraint,anchors,anchorDic,distMat):
    newConstr=[]
    (a,b,t)=constraint
    anchorA=anchors[0]
    anchorB=anchors[0]

    for anch in anchors:
        if(a in anchorDic[anch]):
            anchorA=anch
        if(b in anchorDic[anch]):
            anchorB=anch

    for p in anchorDic[anchorA]:
        if distMat[p][b] >= distMat[a][b]:
            newConstr.append((p,b,t))

    for p in anchorDic[anchorB]:
        if distMat[a][p] >= distMat[a][b]:
            newConstr.append((a,p,t))

    return newConstr


def propagatePairwiseConstraints(constraints,anchors,anchorDic,distMat):
    newConstr=constraints.copy()
    for cons in constraints:
        if cons[2]==1:
            newConstr=newConstr+MLpropagation(cons,anchors,anchorDic,distMat)
        else:
            newConstr=newConstr+CLpropagation(cons,anchors,anchorDic,distMat)

    return newConstr

#------------- Prairwise Prop NPC

def propagatePairwiseConstraintsNPC(constraints,anchors,anchorDic,distMat):
    newConstr=constraints.copy()
    alreadyconstr=[]
    for (p1,p2,t) in constraints:
        if p1 not in alreadyconstr:
             alreadyconstr.append(p1)
        if p2 not in alreadyconstr:
            alreadyconstr.append(p2)

    for cons in constraints:
        if cons[2]==1:
            newConstr=newConstr+MLpropagationNPC(cons,anchors,anchorDic,distMat,alreadyconstr)
        else:
            newConstr=newConstr+CLpropagationNPC(cons,anchors,anchorDic,distMat,alreadyconstr)

    return newConstr


#constraint: a Ml constraint
#anchors: list of anchors
#distMat: euclidean distance matrix
def MLpropagationNPC(constraint,anchors,anchorDic,distMat,alreadyconstr):
    newConstr=[]
    (a,b,t)=constraint

    anchorA=anchors[0]
    minDist=10000
    for anch in anchors:
        if(a in anchorDic[anch]):
            anchorA=anch
    for p in anchorDic[anchorA]:
        if distMat[p][b] <= distMat[a][b] and ( p not in alreadyconstr):
            newConstr.append((p,b,t))
            alreadyconstr.append(p)

    anchorB=anchors[0]
    minDist=10000
    for anch in anchors:
        if(b in anchorDic[anch]):
            anchorB=anch
    for p in anchorDic[anchorB]:
        if distMat[a][p] <= distMat[a][b] and ( p not in alreadyconstr):
            newConstr.append((a,p,t))
            alreadyconstr.append(p)

    return newConstr

#propagate CL constraints
def CLpropagationNPC(constraint,anchors,anchorDic,distMat,alreadyconstr):
    newConstr=[]
    (a,b,t)=constraint
    anchorA=anchors[0]
    anchorB=anchors[0]

    for anch in anchors:
        if(a in anchorDic[anch]):
            anchorA=anch
        if(b in anchorDic[anch]):
            anchorB=anch

    for p in anchorDic[anchorA]:
        if distMat[p][b] >= distMat[a][b] and ( p not in alreadyconstr):
            newConstr.append((p,b,t))
            alreadyconstr.append(p)

    for p in anchorDic[anchorB]:
        if distMat[a][p] >= distMat[a][b] and ( p not in alreadyconstr):
            newConstr.append((a,p,t))
            alreadyconstr.append(p)

    return newConstr
#--------------_____________-----------



#Nr = number of constraint set to generate
#Nc = number of constraints to generate
#data = dataset
#D = DistanceMatrix
#labels = GroundTruth labels of the data
#name: name (including path) of the files that will be created
def createTripletConstraints(Nr,Nc,data,D,labels,name):
    constrSets=[]
    for a in range(0,Nr):
        cons=[]
        for i in range(0,Nc):
            x,y,z=random.sample(range(0, len(labels)), 3)
            while( (x,y,z) in cons or (x,z,y) in cons):
                x,y,z=random.sample(range(0, len(labels)), 3)

            if( (labels[x]==labels[y] and labels[x]==labels[z] and labels[y]==labels[z]) or (labels[x]!=labels[y] and labels[x]!=labels[z]) ): #$P[x]==P[y]==P[z]$ or ($P[x]!=P[y]$ and $P[x]!=P[z]$ and $P[y]!=P[z]$)
                if( D[x][y]<=D[x][z] ):
                    cons.append((x,y,z))
                    #print(labels[x],labels[y],labels[z],D[x][y],D[x][z])
                else:
                    cons.append((x,z,y))
                    #print(labels[x],labels[z],labels[y],D[x][z],D[x][y])
            elif(labels[x]==labels[y] and labels[x]!=labels[z] ):
                cons.append((x,y,z))
                #print(labels[x],labels[y],labels[z])
            elif(labels[x]!=labels[y] and labels[x]==labels[z] ):
                cons.append((x,z,y))
                #print(labels[x],labels[z],labels[y])

        constrSets.append(cons)
        filename=name+str(a)+".txt"
        writeConstraints(filename,cons)
    return constrSets


#Nr = number of constraint set to generate
#Nc = number of constraints to generate
#data = dataset
#D = DistanceMatrix
#labels = GroundTruth labels of the data
#name: name (including path) of the files that will be created
def createInformativeTripletConstraints(data,D,P,labels,name): #TODO
    #constrSets=[]
    for a in range(0,1): #for a in range(0,Nr):
        cons=[]
        infCons=[] #informative constraints
        for i in range(0,50000): #for i in range(0,Nc):
            x,y,z=random.sample(range(0, len(labels)), 3)
            while( (x,y,z) in cons or (x,z,y) in cons):
                x,y,z=random.sample(range(0, len(labels)), 3)

            if( (labels[x]==labels[y] and labels[x]==labels[z] and labels[y]==labels[z]) or (labels[x]!=labels[y] and labels[x]!=labels[z]) ): #$P[x]==P[y]==P[z]$ or ($P[x]!=P[y]$ and $P[x]!=P[z]$ and $P[y]!=P[z]$)
                if( D[x][y]<=D[x][z] ):
                    c=(x,y,z)
                    cons.append(c)
                else:
                    c=(x,z,y)
                    cons.append(c)
            elif(labels[x]==labels[y] and labels[x]!=labels[z] ):
                c=(x,y,z)
                cons.append(c)
            elif(labels[x]!=labels[y] and labels[x]==labels[z] ):
                c=(x,z,y)
                cons.append(c)
            if(not verify_triplet_constraint(P,c)):
                infCons.append(c)

        print("Number of inf constr found: "+str(len(infCons)))
        if(len(infCons)>=200):
            constrSet=random.sample(infCons,200)
        else:
            print("Not enought informative triplets")
            return infCons
        #constrSets.append(constrSet)

        filename=name+str(a)+".txt"
        writeConstraints(filename,constrSet)
        return  constrSet
    #return constrSets

#pw: a set of pairwise constraints in format (point1,point2,type)
#RETURN a set of Triplet constraints generated from the Pairwise constraints. ML(x,y) and CL(x,z) => Triplet(x,y,z)
def createTripletFromPairwise(pw):
    cons=[]
    #verify if at least one ML and one CL constraint are present
    for p1,p2,t1 in pw:
        for p3,p4,t2 in pw:
            if(t1!=t2 and not(p1 in [p3,P4] and p2 in [p3,p4]) ):
                if(p1==p3):
                    if t1==1 and (p1,p2,p4) not in cons:
                        cons.append((p1,p2,p4))
                    elif (p1,p4,p2) not in cons:
                        cons.append((p1,p4,p2))

                elif(p1==p4):
                    if t1==1 and (p2,p1,p3) not in cons:
                        cons.append((p2,p1,p3))
                    elif (p1,p3,p2) not in cons:
                        cons.append((p1,p3,p2))

                elif(p2==p3):
                    if t1==1 and (p2,p1,p4) not in cons:
                        cons.append((p2,p1,p4))
                    elif (p2,p4,p1) not in cons:
                        cons.append((p2,p4,p1))

                elif(p2==p4):
                    if t1==1 and (p2,p1,p3) not in cons:
                        cons.append((p2,p1,p3))
                    elif (p2,p3,p1) not in cons:
                        cons.append((p2,p3,p1))
    return cons

def verify_triplet_constraint(partition,constraint):
    x,y,z = constraint
    if partition[x]==partition[y]:
        return True
    else: #partition[x]!=partition[y]
        if partition[x]!=partition[z]:
            return True
        #else, the constraint i not verified (ex: 0 1 0)
    return False

def verify_triplet_constraints(partition,constraints):
    nVerif=0; nConstr=len(constraints)
    for x,y,z in constraints:
        if partition[x]==partition[y]:
            nVerif+=1
        else: #partition[x]!=partition[y]
            if partition[x]!=partition[z]:
                nVerif+=1
            #else, the constraint i not verified (ex: 0 1 0)
    per=(nVerif*100)/nConstr
    return per

def verify_multiple_tripletSets(partition,constrSets):
    resVer=[]
    for constraints in constrSets:
        resVer.append(verify_triplet_constraints(partition,constraints))
    return resVer


#propagate triplet constraints
def TripletPropagation(triplet,anchors,anchorDic,distMat):
    newConstr=[]
    (a,p,n)=triplet
    anchorA=anchors[0]
    anchorP=anchors[0]
    anchorN=anchors[0]

    for anch in anchors:
        if(a in anchorDic[anch]):
            anchorA=anch
        if(p in anchorDic[anch]):
            anchorP=anch
        if(n in anchorDic[anch]):
            anchorN=anch

    for a2 in anchorDic[anchorA]:
        if distMat[a][p]<= distMat[a2][p]:
            newConstr.append((a2, p, n))
    for p2 in anchorDic[anchorP]:
        if distMat[a][p2] <= distMat[a][p]:
            newConstr.append((a, p2, n))
    for n2 in anchorDic[anchorN]:
        if distMat[a][n2] >= distMat[a][n]:
            newConstr.append((a, p, n2))

    return newConstr

def propagateTripletConstraints(triplets,anchors,anchorDic,distMat):
    newConstr=triplets.copy()
    for triplet in triplets:
        newConstr=newConstr+TripletPropagation(triplet,anchors,anchorDic,distMat)

    return newConstr
