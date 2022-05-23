include "globals.mzn"; 

int: n; % number of points 
int: k_min; 
int: k_max; 

array[1..n, 1..k_max] of float: M; 
array[1..n] of int: P; 
array[1..n] of var 1..k_max: G; 

var min(i in 1..n, k in 1..k_max)(M[i,k]) .. max(i in 1..n, k in 1..k_max)(M[i,k]) : Obj; 

constraint forall (i in 1..n, k in 1..k_max where P[i]=k) ( M[i,k] > Obj -> P[i]=G[i]); 
constraint G[52]!=G[31]; 
constraint G[134]!=G[18]; 
constraint G[85]!=G[139]; 
constraint G[138]!=G[67]; 
constraint G[14]=G[11]; 
constraint G[124]!=G[89]; 
constraint G[22]!=G[122]; 
constraint G[23]!=G[149]; 
constraint G[123]!=G[69]; 
constraint G[90]!=G[50]; 
constraint G[39]!=G[92]; 
constraint G[149]=G[140]; 
constraint G[91]!=G[107]; 
constraint G[128]=G[114]; 
constraint G[4]!=G[113]; 
constraint G[117]=G[114]; 
constraint G[110]=G[113]; 
constraint G[75]!=G[116]; 
constraint G[39]!=G[116]; 
constraint G[118]!=G[25]; 
constraint G[110]!=G[60]; 
constraint G[128]!=G[1]; 
constraint G[35]=G[22]; 
constraint G[33]=G[17]; 
constraint G[111]=G[119]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) minimize Obj; 
output ["G = \(G)\nObj=\(Obj)"]; 
