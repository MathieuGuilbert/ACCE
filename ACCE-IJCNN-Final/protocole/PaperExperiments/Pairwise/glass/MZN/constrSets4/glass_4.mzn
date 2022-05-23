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

constraint G[59]!=G[98]; 
constraint G[109]!=G[45]; 
constraint G[108]!=G[147]; 
constraint G[29]!=G[180]; 
constraint G[36]!=G[124]; 
constraint G[130]=G[97]; 
constraint G[89]!=G[175]; 
constraint G[25]=G[23]; 
constraint G[47]!=G[161]; 
constraint G[211]!=G[148]; 
constraint G[67]!=G[116]; 
constraint G[31]=G[54]; 
constraint G[184]!=G[134]; 
constraint G[133]!=G[175]; 
constraint G[19]!=G[201]; 
constraint G[83]!=G[208]; 
constraint G[159]!=G[55]; 
constraint G[155]!=G[134]; 
constraint G[81]!=G[43]; 
constraint G[7]=G[52]; 
constraint G[133]!=G[64]; 
constraint G[92]!=G[50]; 
constraint G[13]=G[19]; 
constraint G[123]!=G[31]; 
constraint G[17]!=G[81]; 
constraint G[75]!=G[188]; 
constraint G[9]!=G[77]; 
constraint G[171]!=G[150]; 
constraint G[2]=G[61]; 
constraint G[189]!=G[49]; 
constraint G[59]!=G[181]; 
constraint G[175]!=G[45]; 
constraint G[141]!=G[152]; 
constraint G[58]!=G[214]; 
constraint G[184]!=G[38]; 
constraint G[178]!=G[123]; 
constraint G[31]=G[30]; 
constraint G[10]!=G[90]; 
constraint G[149]!=G[2]; 
constraint G[27]=G[23]; 
constraint G[196]!=G[121]; 
constraint G[192]!=G[178]; 
constraint G[187]!=G[55]; 
constraint G[97]!=G[62]; 
constraint G[153]!=G[108]; 
constraint G[137]=G[113]; 
constraint G[81]!=G[192]; 
constraint G[170]!=G[48]; 
constraint G[56]!=G[211]; 
constraint G[171]!=G[148]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) maximize S; 
output ["G = \(G)\nObj=\(S)"]; 
