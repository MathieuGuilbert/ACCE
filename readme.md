# Thesis Mathieu Guilbert
# Projet Involvd
# ACCE

The results presented in the paper are in Devs/protocole/PaperExperiments. (some elements and folders present in them are reminiscence from previous experiments and are not relevant, like the Gecode, Gurobi and MZN folders.)

In the Devs folder, you can find some datasets, a small inventory of the modification done in the ILP approach, a script importing the needed dependancies a folder "protocole" containing our work.

----------------

The folder COP_Kmeans_master contains the implenmentation of the COPKmeans algorithm by B. Babaki found available here https://github.com/Behrouz-Babaki/COP-Kmeans .

The folder ConstrainedClusteringViaPostProcessing_master contains the implementation of the ILP program develloped by Nguyen-Viet-Dung Nghiem et al. for the article Constrained clustering via post-processing, slightly modified to fit our needs.

---------------

To lauch our method, you have to launch the pythonScript.py file with the ID of the desired dataset*, the number of Consensus Partitions to generate (with different base partitions each) and and the number of constraint sets to generate.

In addition, you can provide optional agruments to indicate which method to launch among the following:
'--c', specify if the constrained EAC is to be launched
'--ILP25', specify if ILP is to be launched with 25 percent of the points of each cluster becoming anchors
--ILP10', specify if ILP is to be launched with 10 percent of the points of each cluster becoming anchors
'--ILP5' specify if ILP is to be launched with 5 percent of the points of each cluster becoming anchors
'--cop', specify if COP-Kmeans is to be launched. If one of the 3 previous option is present, then it will launch the associated ILP post-treatment of the results obtained with COP.
'--tripletSimple', specify if triplets constraints are to be used in the postprocess instead of MLs and CLs (not compatible with '--cop')

Other options are available, but the either represents non-functional methods or simply methods that were not presented in the paper.

The results of the program are put in the 'Architecture' folder.

------------

*IDs of the datasets: 0=halfmoon , 1=ds2c2sc13 with 13 clusters , 2=complex9 , 3=iris , 4=glass, 5=ionosphere , 6=11, a small handcrafted 2 dataset of only 11 points , 7=ds2c2sc13 with 2 clusters , 8=ds2c2sc13 with 5 clusters.
