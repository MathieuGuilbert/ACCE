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

constraint G[51]!=G[34]; 
constraint G[42]!=G[75]; 
constraint G[42]!=G[24]; 
constraint G[133]!=G[124]; 
constraint G[122]!=G[106]; 
constraint G[28]!=G[9]; 
constraint G[8]!=G[89]; 
constraint G[42]!=G[21]; 
constraint G[91]!=G[53]; 
constraint G[119]!=G[70]; 
constraint G[122]!=G[20]; 
constraint G[62]!=G[19]; 
constraint G[111]!=G[33]; 
constraint G[46]!=G[116]; 
constraint G[14]!=G[8]; 
constraint G[14]!=G[8]; 
constraint G[13]!=G[84]; 
constraint G[127]!=G[125]; 
constraint G[76]!=G[112]; 
constraint G[106]!=G[49]; 
constraint G[1]!=G[106]; 
constraint G[39]!=G[98]; 
constraint G[117]!=G[64]; 
constraint G[56]!=G[132]; 
constraint G[108]!=G[114]; 
constraint G[110]!=G[68]; 
constraint G[130]!=G[56]; 
constraint G[95]!=G[54]; 
constraint G[6]!=G[67]; 
constraint G[120]!=G[42]; 
constraint G[77]!=G[51]; 
constraint G[12]!=G[14]; 
constraint G[21]!=G[64]; 
constraint G[18]!=G[102]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) maximize S; 
output ["G = \(G)\nObj=\(S)"]; 
