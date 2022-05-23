include "globals.mzn"; 

int: n; % number of points 
int: k_min; 
int: k_max; 

array[1..n, 1..k_max] of int: M; 
array[1..n] of int: P; 
array[1..n] of var 1..k_max: G; 

var min(M) .. max(M) : Obj; 

constraint forall (i in 1..n, k in 1..k_max where P[i]=k) ( M[i,k] > Obj -> P[i]=G[i]); 


constraint G[237]!=G[1303]; 
constraint G[758]=G[787]; 
constraint G[2234]!=G[1146]; 
constraint G[1341]=G[1318]; 
constraint G[1358]!=G[835]; 
constraint G[1823]=G[1495]; 
constraint G[2857]=G[2418]; 
constraint G[2438]!=G[1455]; 
constraint G[17]!=G[1938]; 
constraint G[411]!=G[2613]; 
constraint G[443]!=G[249]; 
constraint G[202]!=G[853]; 
constraint G[2414]!=G[749]; 
constraint G[860]!=G[2681]; 
constraint G[2398]=G[2528]; 
constraint G[276]!=G[2655]; 
constraint G[827]!=G[2795]; 
constraint G[2622]!=G[230]; 
constraint G[1935]!=G[948]; 
constraint G[2707]!=G[2100]; 
constraint G[2956]!=G[732]; 
constraint G[244]!=G[1227]; 
constraint G[274]!=G[1716]; 
constraint G[166]!=G[451]; 
constraint G[1902]!=G[742]; 
constraint G[2352]!=G[305]; 
constraint G[2692]!=G[1403]; 
constraint G[1396]!=G[670]; 
constraint G[1104]=G[1862]; 
constraint G[2608]!=G[702]; 
constraint G[2122]!=G[369]; 
constraint G[538]=G[661]; 
constraint G[2830]!=G[722]; 
constraint G[665]!=G[1633]; 
constraint G[935]!=G[568]; 
constraint G[168]!=G[2430]; 
constraint G[2251]!=G[2668]; 
constraint G[1098]!=G[1000]; 
constraint G[2336]!=G[55]; 
constraint G[248]!=G[39]; 
constraint G[1696]!=G[2835]; 
constraint G[2606]!=G[531]; 
constraint G[2007]!=G[646]; 
constraint G[2781]=G[2478]; 
constraint G[2314]=G[2299]; 
constraint G[1561]!=G[710]; 
constraint G[1365]=G[1768]; 
constraint G[2991]!=G[1176]; 
constraint G[1712]!=G[2825]; 
constraint G[246]!=G[2835]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) minimize Obj; 
output ["G = \(G)\nObj=\(Obj)"]; 
