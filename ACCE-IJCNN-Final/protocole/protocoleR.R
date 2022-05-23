#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)
#args should be datafile, ml file, cl file, k, outputFile

# test if there are 5 arguments
if (length(args)!=5) {
  stop("5 arguments must be specified", call.=FALSE)
} 
datafile=args[1]
MLfile=args[2]
CLfile=args[3]
k=as.integer(args[4])
outputfile=args[5]
#datafile="/home/mathieu/Documents/Doctorat/Stage/git/stage-involvd-mathieu-guilbert-2021/Devs/protocole/Architecture/halfmoon/halfmoon.txt"
#MLfile="/home/mathieu/Documents/Doctorat/Stage/git/stage-involvd-mathieu-guilbert-2021/Devs/protocole/Architecture/halfmoon/constraints/halfmoonML0.txt"
#CLfile="/home/mathieu/Documents/Doctorat/Stage/git/stage-involvd-mathieu-guilbert-2021/Devs/protocole/Architecture/halfmoon/constraints/halfmoonCL0.txt"
#k="2"
#outputfile="/home/mathieu/Documents/Doctorat/Stage/git/stage-involvd-mathieu-guilbert-2021/Devs/protocole/Architecture/halfmoon/constraints/outputFirstMPCK.txt"

#needed libraries
library(conclust)

#Read the constraints
processConstraintFile = function(MLpath,CLpath) {
  #print("Constraints")
  ml=as.matrix(read.table(MLpath))
  cl=as.matrix(read.table(CLpath))
  return(list("ml" = ml, "cl" = cl))
}

#Read the data
readData = function(dataPath) {
  #print("Data")
  d=as.matrix(read.table(dataPath))
  return(d)
}

constraints=processConstraintFile(MLfile,CLfile)
#print(constraints)

data=readData(datafile)
#print(data)

print("call MPCKM")
metric = mpckm(data, k, constraints$ml, constraints$cl)
#print(metric)
write.table(metric, outputfile, quote=F)
#print("-----")

#print("call COPKM")
#cop = ckmeans(data, k, constraints$ml, constraints$cl)
#print(cop)
#write.table(cop, outputfile, quote=F)
#print("-----")
