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

constraint G[3]!=G[20]; 
constraint G[4]!=G[21]; 
constraint G[4]!=G[22]; 
constraint G[23]!=G[24]; 
constraint G[25]!=G[26]; 
constraint G[27]!=G[5]; 
constraint G[6]!=G[28]; 
constraint G[4]!=G[29]; 
constraint G[30]!=G[7]; 
constraint G[31]!=G[32]; 
constraint G[25]!=G[33]; 
constraint G[8]!=G[34]; 
constraint G[35]!=G[36]; 
constraint G[9]!=G[37]; 
constraint G[2]!=G[6]; 
constraint G[2]!=G[6]; 
constraint G[10]!=G[38]; 
constraint G[11]!=G[39]; 
constraint G[40]!=G[41]; 
constraint G[26]!=G[12]; 
constraint G[13]!=G[26]; 
constraint G[14]!=G[42]; 
constraint G[43]!=G[15]; 
constraint G[16]!=G[44]; 
constraint G[45]!=G[46]; 
constraint G[47]!=G[48]; 
constraint G[49]!=G[16]; 
constraint G[50]!=G[17]; 
constraint G[18]!=G[51]; 
constraint G[52]!=G[4]; 
constraint G[53]!=G[3]; 
constraint G[19]!=G[2]; 
constraint G[29]!=G[15]; 
constraint G[54]!=G[55]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) maximize S; 
output ["G = \(G)\nObj=\(S)"]; 
