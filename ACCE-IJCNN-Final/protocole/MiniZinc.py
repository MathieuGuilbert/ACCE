from minizinc import Driver, Model, Solver, default_driver, find_driver

#convert a distance matrix into a dzn file
#name is the intended name of the dzn file
#data is a list of list
#n is the number of elements
#k_min and k_max are the limits of the number of clusters
def listToDZN(name,data,n,k_min,k_max):
    #write DZN
    f = open(name+".dzn", "w")
    f.write("n="+str(n)+";")
    f.write("\nk_min="+str(k_min)+";")
    f.write("\nk_max="+str(k_max)+";")
    #for the first line
    firstline="\ndist=[| "
    for i in range(len(data[0])):
            value=str(data[0][i])
            firstline+=value
            if(not (i==len(data[0])-1 ) ):
                firstline+=", "
    f.write(firstline)
    #for all the other lines
    for j in range(1,len(data)):
        line="\n | "
        for i in range(len(data[0])):
            value=str(data[j][i])
            line+=value
            if(not (i==len(data[0])-1 ) ):
                line+=", "
        f.write(line)
    f.write(" |];")
    f.close()
    return

def writeMZN(path,name,constraints):
    f = open(path+"/"+name+".mzn", "w")
    f.write("include \"globals.mzn\"; \n")
    f.write("\n")
    f.write("int: n; % number of points \n")
    f.write("int: k_min; \n")
    f.write("int: k_max; \n")
    f.write("\n")
    f.write("array[1..n, 1..n] of float: dist; \n")
    f.write("array[1..n] of var 1..k_max: G; \n")
    f.write("\n")
    f.write("var min(i,j in 1..n where i<j)(dist[i,j]) .. max(i,j in 1..n where i<j)(dist[i,j]) : S; \n")
    f.write("\n")
    f.write("constraint forall (i,j in 1..n where i<j) ( dist[i,j] < S -> G[i] == G[j] ); \n")
    f.write("constraint G[1] = 1; \n")
    f.write("constraint value_precede_chain([i | i in 1..k_max], G); \n")
    f.write("constraint max(G) >= k_min; \n")
    f.write("\n")
    for (p1,p2,t) in constraints:
        if t==1: #1 -> ML
            f.write("constraint G["+str(p1+1)+"]=G["+str(p2+1)+"]; \n")
        if t==-1: #-1 -> CL
            f.write("constraint G["+str(p1+1)+"]!=G["+str(p2+1)+"]; \n")
    #constraints
    f.write("%%%%%%%%%%%%% \n")
    f.write("\n")
    f.write("solve ::int_search(G, first_fail, indomain_min, complete) maximize S; \n")
    f.write("output [\"G = \(G)\\nObj=\(S)\"]; \n")
    return

#same as writeMZN but only writing the CL in constraints (not the ML)
def writeMZNonlyCL(path,name,constraints):
    f = open(path+"/"+name+".mzn", "w")
    f.write("include \"globals.mzn\"; \n")
    f.write("\n")
    f.write("int: n; % number of points \n")
    f.write("int: k_min; \n")
    f.write("int: k_max; \n")
    f.write("\n")
    f.write("array[1..n, 1..n] of float: dist; \n")
    f.write("array[1..n] of var 1..k_max: G; \n")
    f.write("\n")
    f.write("var min(i,j in 1..n where i<j)(dist[i,j]) .. max(i,j in 1..n where i<j)(dist[i,j]) : S; \n")
    f.write("\n")
    f.write("constraint forall (i,j in 1..n where i<j) ( dist[i,j] < S -> G[i] == G[j] ); \n")
    f.write("constraint G[1] = 1; \n")
    f.write("constraint value_precede_chain([i | i in 1..k_max], G); \n")
    f.write("constraint max(G) >= k_min; \n")
    f.write("\n")
    for (p1,p2,t) in constraints:
        if t==-1: #-1 -> CL
            f.write("constraint G["+str(p1+1)+"]!=G["+str(p2+1)+"]; \n")
    #constraints
    f.write("%%%%%%%%%%%%% \n")
    f.write("\n")
    f.write("solve ::int_search(G, first_fail, indomain_min, complete) maximize S; \n")
    f.write("output [\"G = \(G)\\nObj=\(S)\"]; \n")
    return

def DFS(repr, start, graph, cc):
    for u in (graph[start]): #Pour chaque u dans graphe[départ]
        if cc[u] == -1 :
            cc[u] = repr
            DFS(repr,u,graph,cc)

#distMat: a distance matrix (list of list of numbers)
#constraints: list of constraints (if it contains any CL they are ignored)
#Alternative way to obtain: the distance matrix modified according to the constraint and a dictionnary discribing wich data elements are combined
#       , a list of updated CL
#       , a dictionnary
def aggregationML(distMat,constraints):
    n=len(distMat)
    ML=[]
    CL=[]
    for e1,e2,t in constraints:
        if t==-1:
            CL.append((e1,e2,t))
        else:
            ML.append((e1,e2,t))

    #Create a graph using the MLs #tous les points des données sont dans le graphe, edges=ML
    adj=[]
    cc=[] #composantes connexes
    for i in range(0,n):
        adj.append([]) #Pour chaque point de 1 à n créer une liste d’adjacence vide` #Doit contenir element principal ?
        cc.append(-1) #Initialiser cc[i]=-1 pour tous les éléments
    #print(cc)

    for o1,o2,t in ML:
            adj[o1].append(o2) #Ajouter o1 à la liste d’adjacence de o2
            adj[o2].append(o1) #Ajouter o2 à la liste d’adjacence de o1
    #print(adj)

    #Recherche des composantes connexes du graphe :
    ind=0
    for i in range(0,n):
        if cc[i]==-1:
            cc[i]=ind
            DFS(ind,i,adj,cc) #dfs=parcours profondeur, chaque fois qu’on touche un point p cc[p] devient i
            ind=ind+1

    #Créer un tableau SP de superpoints où un superpoint est une composante connexe (donc une liste de points)
    SP=[]
    for i in range(0,max(cc)+1): #init: create the empty superpoints
        SP.append([])
    for i in range(0,n): #add the point to their corresponding SP
        SP[cc[i]].append(i)
    #print("SP: "+str(SP))

    #Créer un tableau inSP donnant pour chaque point de 0 à n-1 le numéro du super point auquel il appartient
    inSP=cc.copy()

    #Calcul nouvelle Matrice de distance dSP
    dSP=[] #FAUT INITIALISER.
    maxDistMat=max([sublist[-1] for sublist in distMat])
    for i in range (0,len(SP)):
        dSP.append([])
        for j in range (0,len(SP)):
            dSP[i].append(maxDistMat)

    for i in range (0,len(SP)):
        for j in range (0,len(SP)):
            #dSP[i][j]=max([sublist[-1] for sublist in distMat]) # remplacer par le max des distances si distance non normalisée
            for u in SP[i]:
                for v in SP[j]:
                    dSP[i][j]= min(distMat[u][v],dSP[i][j])


    #Calcul des nouvelles CL
    newCL=[]
    for o1,o2,t in CL:
        newCL.append((inSP[o1],inSP[o2],t))

    return dSP,newCL,inSP

#Reverse the aggregation process
def reverseAggregation(inSP,labels,partition):
    desagr=[]
    for i in range(len(labels)):
        #search i in inSP
        desagr.append(partition[inSP[i]])
        #print(desagr)
    return desagr

def testAggregation():
    print("Test aggregation")
    distMat=[[0.0,3.3,1.1,1.6,4.0,4.4,3.3],
            [3.3,0.0,2.4,3.3,1.6,2.0,3.4],
            [1.1,2.4,0.0,1.1,2.9,2.9,2.4],
            [1.6,3.3,1.1,0.0,3.3,2.3,1.8],
            [4.0,1.6,2.9,3.3,0.0,1.1,2.6],
            [4.0,2.0,2.9,2.3,1.1,0.0,1.6],
            [3.3,3.4,2.4,1.8,2.6,1.6,0.0]]

    constraints=[(0,2,1),(3,2,1),(4,5,1),(2,6,-1)]

    labels=[0,1,0,0,1,1,1]

    dSP,newCL,inSP=aggregationML(distMat,constraints)

    print("newDistMat "+str(dSP))
    print("newCL "+str(newCL))
    print(inSP)

    print()
    print(reverseAggregation(inSP,labels,[0,1,1,1]))

    #print(reversePreTreatment([1,0,2,3],inSP,labels,[1,0,1,1]))

#Function from the internship, not used anymore. Was originaly in pythonScript
#lauches a minizinc process using a given mzn file and a dzn file
#type: 1 if default, 2 if aggreg, 3 if reorder1, 4 if reorder2
def launchMiniZinc(mznPath,dznPath,resultPath,type,labels,inSP,newOrder):
    print("lauch mzn")
    if(type not in [1,2,3,4]):
        print("ERROR TYPE")

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

    #transformation string resultat en liste
    part=exctractList(str(result))
    ari=0
    if(type==1):
        ari=metrics.adjusted_rand_score(labels,part)
    elif(type==2):
        rev=reverseAggregation(inSP,labels,part)
        ari=metrics.adjusted_rand_score(labels,rev)
    else:
        #ari=metrics.adjusted_rand_score(labels,reversePreTreatment(newOrder,inSP,labels,part))
        rev=reverseOrder(newOrder,inSP,part)
        #print("len labels "+str(len(labels))+" ,len rev "+str(len(rev)))
        ari=metrics.adjusted_rand_score(labels,rev)
    print("ARI="+str(ari))

    writeResMZN(resultPath,result,elapsed,ari,rev)
    print("finish mzn")

#redefine the location of MiniZinc
def DefineMinizincDriver():
    print("start redefine MZN python")
    #WARNING: my_driver takes a list !
    my_driver = find_driver(["/home/mathieu/Documents/MiniZinc/bin"]) #WARNING: it takes a list !
    print("driver")
    print(my_driver)
    my_driver.make_default()
