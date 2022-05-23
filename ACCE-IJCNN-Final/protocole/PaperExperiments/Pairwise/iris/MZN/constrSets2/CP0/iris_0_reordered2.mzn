include "globals.mzn"; 

int: n; % number of points 
int: k_min; 
int: k_max; 

array[1..n, 1..n] of float: dist; 
array[1..n] of var 1..k_max: G; 

var min(i,j in 1..n where i<j)(dist[i,j]) .. max(i,j in 1..n where i<j)(dist[i,j]) : S; 

constraint forall (i,j in 1..n where i<j) ( dist[i,j] < S -> G[i] == G[j] ); 
constraint G[1] = 1; 
constraint value_precede_chain([i | i in 1..k_max], G); 
constraint max(G) >= k_min; 

constraint G[3]!=G[18]; 
constraint G[19]!=G[4]; 
constraint G[20]!=G[1]; 
constraint G[2]!=G[21]; 
constraint G[5]!=G[22]; 
constraint G[6]!=G[23]; 
constraint G[7]!=G[24]; 
constraint G[23]!=G[25]; 
constraint G[26]!=G[27]; 
constraint G[28]!=G[29]; 
constraint G[30]!=G[8]; 
constraint G[31]!=G[9]; 
constraint G[10]!=G[30]; 
constraint G[32]!=G[11]; 
constraint G[25]!=G[12]; 
constraint G[13]!=G[33]; 
constraint G[34]!=G[27]; 
constraint G[25]!=G[35]; 
constraint G[28]!=G[14]; 
constraint G[36]!=G[33]; 
constraint G[30]!=G[37]; 
constraint G[15]!=G[38]; 
constraint G[39]!=G[40]; 
constraint G[41]!=G[36]; 
constraint G[42]!=G[3]; 
constraint G[43]!=G[16]; 
constraint G[44]!=G[17]; 
constraint G[11]!=G[45]; 
constraint G[46]!=G[32]; 
constraint G[47]!=G[48]; 
constraint G[38]!=G[49]; 
constraint G[23]!=G[50]; 
constraint G[8]!=G[2]; 
constraint G[47]!=G[51]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) maximize S; 
output ["G = \(G)\nObj=\(S)"]; 
