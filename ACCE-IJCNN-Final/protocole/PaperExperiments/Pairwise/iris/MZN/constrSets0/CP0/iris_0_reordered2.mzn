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

constraint G[29]!=G[114]; 
constraint G[10]!=G[121]; 
constraint G[126]!=G[118]; 
constraint G[126]!=G[96]; 
constraint G[49]!=G[137]; 
constraint G[91]!=G[42]; 
constraint G[9]!=G[26]; 
constraint G[115]!=G[101]; 
constraint G[1]!=G[64]; 
constraint G[114]!=G[80]; 
constraint G[93]!=G[47]; 
constraint G[42]!=G[85]; 
constraint G[59]!=G[55]; 
constraint G[64]!=G[125]; 
constraint G[59]!=G[101]; 
constraint G[80]!=G[60]; 
constraint G[94]!=G[24]; 
constraint G[128]!=G[77]; 
constraint G[83]!=G[28]; 
constraint G[68]!=G[140]; 
constraint G[118]!=G[1]; 
constraint G[128]!=G[19]; 
constraint G[87]!=G[11]; 
constraint G[35]!=G[8]; 
constraint G[109]!=G[27]; 
constraint G[110]!=G[4]; 
constraint G[23]!=G[128]; 
constraint G[94]!=G[54]; 
constraint G[73]!=G[41]; 
constraint G[78]!=G[47]; 
constraint G[37]!=G[53]; 
constraint G[144]!=G[114]; 
constraint G[96]!=G[16]; 
constraint G[53]!=G[106]; 
constraint G[96]!=G[113]; 
constraint G[120]!=G[59]; 
constraint G[131]!=G[21]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) maximize S; 
output ["G = \(G)\nObj=\(S)"]; 
