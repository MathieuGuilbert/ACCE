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

constraint G[32]!=G[118]; 
constraint G[124]!=G[7]; 
constraint G[103]!=G[15]; 
constraint G[75]!=G[41]; 
constraint G[38]!=G[53]; 
constraint G[28]!=G[127]; 
constraint G[18]!=G[98]; 
constraint G[127]!=G[82]; 
constraint G[78]!=G[49]; 
constraint G[73]!=G[62]; 
constraint G[105]!=G[22]; 
constraint G[46]!=G[54]; 
constraint G[29]!=G[105]; 
constraint G[63]!=G[4]; 
constraint G[82]!=G[20]; 
constraint G[11]!=G[61]; 
constraint G[89]!=G[49]; 
constraint G[82]!=G[130]; 
constraint G[73]!=G[3]; 
constraint G[116]!=G[61]; 
constraint G[105]!=G[123]; 
constraint G[12]!=G[132]; 
constraint G[101]!=G[48]; 
constraint G[70]!=G[116]; 
constraint G[74]!=G[32]; 
constraint G[109]!=G[8]; 
constraint G[125]!=G[27]; 
constraint G[4]!=G[51]; 
constraint G[84]!=G[63]; 
constraint G[50]!=G[108]; 
constraint G[132]!=G[128]; 
constraint G[127]!=G[77]; 
constraint G[22]!=G[75]; 
constraint G[50]!=G[83]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) maximize S; 
output ["G = \(G)\nObj=\(S)"]; 
