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

constraint G[3]!=G[21]; 
constraint G[22]!=G[23]; 
constraint G[24]!=G[25]; 
constraint G[4]!=G[26]; 
constraint G[27]!=G[5]; 
constraint G[2]!=G[28]; 
constraint G[6]!=G[2]; 
constraint G[29]!=G[30]; 
constraint G[7]!=G[31]; 
constraint G[8]!=G[32]; 
constraint G[22]!=G[9]; 
constraint G[5]!=G[33]; 
constraint G[34]!=G[2]; 
constraint G[35]!=G[10]; 
constraint G[36]!=G[11]; 
constraint G[37]!=G[38]; 
constraint G[39]!=G[12]; 
constraint G[40]!=G[13]; 
constraint G[41]!=G[14]; 
constraint G[42]!=G[43]; 
constraint G[32]!=G[44]; 
constraint G[45]!=G[46]; 
constraint G[2]!=G[47]; 
constraint G[15]!=G[48]; 
constraint G[49]!=G[25]; 
constraint G[50]!=G[16]; 
constraint G[51]!=G[52]; 
constraint G[53]!=G[54]; 
constraint G[42]!=G[55]; 
constraint G[56]!=G[27]; 
constraint G[47]!=G[2]; 
constraint G[17]!=G[37]; 
constraint G[57]!=G[18]; 
constraint G[19]!=G[58]; 
constraint G[20]!=G[59]; 
constraint G[1]!=G[50]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) maximize S; 
output ["G = \(G)\nObj=\(S)"]; 
