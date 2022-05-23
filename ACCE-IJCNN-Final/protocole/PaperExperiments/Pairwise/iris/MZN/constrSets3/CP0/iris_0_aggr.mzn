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

constraint G[18]!=G[77]; 
constraint G[36]!=G[81]; 
constraint G[36]!=G[79]; 
constraint G[103]!=G[55]; 
constraint G[97]!=G[65]; 
constraint G[48]!=G[25]; 
constraint G[2]!=G[123]; 
constraint G[36]!=G[78]; 
constraint G[128]!=G[15]; 
constraint G[121]!=G[74]; 
constraint G[97]!=G[76]; 
constraint G[44]!=G[69]; 
constraint G[115]!=G[75]; 
constraint G[20]!=G[133]; 
constraint G[49]!=G[2]; 
constraint G[49]!=G[2]; 
constraint G[13]!=G[107]; 
constraint G[11]!=G[61]; 
constraint G[82]!=G[104]; 
constraint G[65]!=G[8]; 
constraint G[1]!=G[65]; 
constraint G[9]!=G[113]; 
constraint G[114]!=G[27]; 
constraint G[32]!=G[80]; 
constraint G[67]!=G[127]; 
constraint G[101]!=G[73]; 
constraint G[99]!=G[32]; 
constraint G[106]!=G[16]; 
constraint G[30]!=G[57]; 
constraint G[122]!=G[36]; 
constraint G[87]!=G[18]; 
constraint G[41]!=G[49]; 
constraint G[78]!=G[27]; 
constraint G[60]!=G[111]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) maximize S; 
output ["G = \(G)\nObj=\(S)"]; 
