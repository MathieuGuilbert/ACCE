#This file allows to verifiy properties on previously obtained results.

#ILP code
from ConstrainedClusteringViaPostProcessing_master.models.pw_csize_ilp import run_modified_model

import os
import time
import argparse
from constraints import *
from data import *

GroundTruth=[HalfmoonLabels,ds2labels13,complex9labels,irisLabels,glassLabels,ionosphereLabels,[0,0,0,0,1,1,1,1,0,0,0],ds2labels2,ds2labels5]

#read a given Allocation Matrix
def readMatAlloc(matAllocFile):
    mat= []
    with open(matAllocFile, 'r') as f:
        for line in f:
            line = line.strip()
            if line != '':
                part=eval(line)
                mat.append(part)
    return mat

#--------------

#Verify the execution time of a the ILP process given a dataset, a constraint file and and an allocation matrix file
def verifILPexecTime(datasetID,consfile,matAllocFile):
    i=int(datasetID)

    constraints=read_constraints(consfile)
    CL,ML=extractCLandML3D(constraints)

    MatAlloc=readMatAlloc(matAllocFile)[0]
    n=len(MatAlloc)
    k=len(MatAlloc[0])

    start_time = time.time()
    ILP_res=run_modified_model(n,k,MatAlloc,ML,CL,GroundTruth[i])
    execTime=(time.time() - start_time)
    print(ILP_res)
    print("--- %s seconds ---" % execTime)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run ILP algorithm')
    parser.add_argument('dataset', help='int indicating the dataset to test', type=int)
    parser.add_argument('consfile', help='number of consensus partition to generate', type=str)
    parser.add_argument('matAllocFile', help='number of constraint sets to generate', type=str)
    args = parser.parse_args()

    verifILPexecTime(args.dataset,args.consfile,args.matAllocFile)
