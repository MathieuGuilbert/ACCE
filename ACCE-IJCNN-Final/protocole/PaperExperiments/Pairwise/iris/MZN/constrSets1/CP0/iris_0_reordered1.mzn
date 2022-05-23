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

constraint G[57]!=G[117]; 
constraint G[140]!=G[5]; 
constraint G[122]!=G[127]; 
constraint G[65]!=G[47]; 
constraint G[24]!=G[87]; 
constraint G[137]!=G[92]; 
constraint G[136]!=G[20]; 
constraint G[124]!=G[24]; 
constraint G[107]!=G[122]; 
constraint G[14]!=G[2]; 
constraint G[106]!=G[21]; 
constraint G[90]!=G[116]; 
constraint G[24]!=G[107]; 
constraint G[74]!=G[45]; 
constraint G[7]!=G[117]; 
constraint G[131]!=G[59]; 
constraint G[33]!=G[64]; 
constraint G[2]!=G[3]; 
constraint G[73]!=G[103]; 
constraint G[30]!=G[138]; 
constraint G[122]!=G[40]; 
constraint G[44]!=G[20]; 
constraint G[84]!=G[111]; 
constraint G[79]!=G[97]; 
constraint G[100]!=G[44]; 
constraint G[123]!=G[107]; 
constraint G[26]!=G[43]; 
constraint G[130]!=G[79]; 
constraint G[135]!=G[64]; 
constraint G[133]!=G[129]; 
constraint G[22]!=G[89]; 
constraint G[136]!=G[55]; 
constraint G[86]!=G[110]; 
constraint G[30]!=G[136]; 
constraint G[30]!=G[110]; 
constraint G[150]!=G[15]; 
constraint G[33]!=G[61]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) maximize S; 
output ["G = \(G)\nObj=\(S)"]; 
