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

constraint G[106]!=G[39]; 
constraint G[29]!=G[106]; 
constraint G[5]!=G[117]; 
constraint G[93]!=G[12]; 
constraint G[58]!=G[7]; 
constraint G[146]=G[116]; 
constraint G[17]!=G[117]; 
constraint G[57]!=G[37]; 
constraint G[122]=G[108]; 
constraint G[71]!=G[49]; 
constraint G[62]!=G[121]; 
constraint G[120]!=G[87]; 
constraint G[17]!=G[87]; 
constraint G[62]!=G[32]; 
constraint G[13]!=G[125]; 
constraint G[54]!=G[128]; 
constraint G[8]=G[29]; 
constraint G[80]!=G[107]; 
constraint G[12]=G[48]; 
constraint G[85]!=G[150]; 
constraint G[71]=G[78]; 
constraint G[133]=G[142]; 
constraint G[9]!=G[128]; 
constraint G[21]!=G[114]; 
constraint G[93]=G[89]; 
constraint G[43]!=G[97]; 
constraint G[3]!=G[136]; 
constraint G[134]=G[114]; 
constraint G[39]=G[12]; 
constraint G[64]!=G[35]; 
constraint G[4]!=G[147]; 
constraint G[107]!=G[99]; 
constraint G[33]!=G[134]; 
constraint G[22]!=G[109]; 
constraint G[12]!=G[83]; 
constraint G[107]!=G[32]; 
constraint G[76]!=G[135]; 
constraint G[76]!=G[4]; 
constraint G[118]!=G[45]; 
constraint G[22]!=G[62]; 
constraint G[56]!=G[107]; 
constraint G[48]=G[30]; 
constraint G[12]!=G[145]; 
constraint G[135]=G[112]; 
constraint G[102]!=G[64]; 
constraint G[46]!=G[129]; 
constraint G[73]!=G[19]; 
constraint G[135]!=G[66]; 
constraint G[71]!=G[108]; 
constraint G[132]=G[134]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) maximize S; 
output ["G = \(G)\nObj=\(S)"]; 
