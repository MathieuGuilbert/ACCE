include "globals.mzn"; 

int: n; % number of points 
int: k_min; 
int: k_max; 

array[1..n, 1..k_max] of float: M; 
array[1..n] of int: P; 
array[1..n] of var 1..k_max: G; 

var min(i in 1..n, k in 1..k_max)(M[i,k]) .. max(i in 1..n, k in 1..k_max)(M[i,k]) : Obj; 

constraint forall (i in 1..n, k in 1..k_max where P[i]=k) ( M[i,k] > Obj -> P[i]=G[i]); 
constraint G[31]=G[50]; 
constraint G[129]!=G[73]; 
constraint G[18]!=G[67]; 
constraint G[5]!=G[138]; 
constraint G[7]!=G[93]; 
constraint G[143]=G[125]; 
constraint G[33]!=G[124]; 
constraint G[86]!=G[43]; 
constraint G[46]!=G[109]; 
constraint G[89]!=G[32]; 
constraint G[2]!=G[76]; 
constraint G[15]!=G[96]; 
constraint G[118]!=G[7]; 
constraint G[99]=G[68]; 
constraint G[122]!=G[49]; 
constraint G[91]=G[89]; 
constraint G[4]=G[15]; 
constraint G[60]=G[95]; 
constraint G[65]=G[81]; 
constraint G[45]=G[18]; 
constraint G[29]=G[26]; 
constraint G[110]!=G[38]; 
constraint G[117]!=G[5]; 
constraint G[100]!=G[9]; 
constraint G[41]!=G[89]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) minimize Obj; 
output ["G = \(G)\nObj=\(Obj)"]; 
