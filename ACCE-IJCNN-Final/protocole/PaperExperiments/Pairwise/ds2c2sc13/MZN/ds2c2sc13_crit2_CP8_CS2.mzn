include "globals.mzn"; 

int: n; % number of points 
int: k_min; 
int: k_max; 

array[1..n, 1..k_max] of int: M; 
array[1..n] of int: P; 
array[1..n] of var 1..k_max: G; 

var min(M) .. max(M) : Obj; 

constraint forall (i in 1..n, k in 1..k_max where P[i]=k) ( M[i,k] > Obj -> P[i]=G[i]); 


constraint G[27]!=G[121]; 
constraint G[398]!=G[317]; 
constraint G[423]!=G[17]; 
constraint G[487]!=G[452]; 
constraint G[322]!=G[404]; 
constraint G[123]!=G[283]; 
constraint G[540]!=G[251]; 
constraint G[504]!=G[372]; 
constraint G[191]!=G[405]; 
constraint G[318]!=G[469]; 
constraint G[579]!=G[505]; 
constraint G[523]=G[535]; 
constraint G[466]!=G[43]; 
constraint G[210]!=G[130]; 
constraint G[204]!=G[28]; 
constraint G[110]!=G[455]; 
constraint G[353]!=G[9]; 
constraint G[79]=G[107]; 
constraint G[191]!=G[423]; 
constraint G[571]!=G[389]; 
constraint G[520]!=G[321]; 
constraint G[420]!=G[490]; 
constraint G[230]!=G[352]; 
constraint G[408]!=G[378]; 
constraint G[210]!=G[151]; 
constraint G[13]!=G[281]; 
constraint G[183]=G[173]; 
constraint G[127]!=G[287]; 
constraint G[532]!=G[554]; 
constraint G[564]!=G[172]; 
constraint G[228]!=G[500]; 
constraint G[106]!=G[398]; 
constraint G[468]!=G[525]; 
constraint G[67]!=G[500]; 
constraint G[456]!=G[155]; 
constraint G[448]!=G[485]; 
constraint G[325]!=G[454]; 
constraint G[587]!=G[454]; 
constraint G[265]!=G[52]; 
constraint G[570]!=G[289]; 
constraint G[386]!=G[6]; 
constraint G[203]!=G[588]; 
constraint G[385]!=G[498]; 
constraint G[466]!=G[237]; 
constraint G[532]!=G[316]; 
constraint G[468]!=G[29]; 
constraint G[213]!=G[49]; 
constraint G[5]=G[88]; 
constraint G[322]!=G[480]; 
constraint G[84]!=G[396]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) minimize Obj; 
output ["G = \(G)\nObj=\(Obj)"]; 
