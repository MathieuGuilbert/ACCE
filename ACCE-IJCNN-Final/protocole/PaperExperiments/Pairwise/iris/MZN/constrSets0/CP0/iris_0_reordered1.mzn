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

constraint G[122]!=G[94]; 
constraint G[16]!=G[92]; 
constraint G[127]!=G[41]; 
constraint G[125]!=G[101]; 
constraint G[1]!=G[137]; 
constraint G[122]!=G[10]; 
constraint G[111]!=G[4]; 
constraint G[103]!=G[29]; 
constraint G[4]!=G[81]; 
constraint G[133]!=G[128]; 
constraint G[146]!=G[62]; 
constraint G[23]!=G[130]; 
constraint G[72]!=G[124]; 
constraint G[15]!=G[79]; 
constraint G[55]!=G[97]; 
constraint G[86]!=G[101]; 
constraint G[55]!=G[87]; 
constraint G[55]!=G[145]; 
constraint G[143]!=G[117]; 
constraint G[64]!=G[40]; 
constraint G[145]!=G[135]; 
constraint G[125]!=G[50]; 
constraint G[54]!=G[145]; 
constraint G[126]!=G[103]; 
constraint G[97]!=G[140]; 
constraint G[60]!=G[74]; 
constraint G[110]!=G[34]; 
constraint G[140]!=G[110]; 
constraint G[139]!=G[146]; 
constraint G[92]!=G[130]; 
constraint G[38]!=G[13]; 
constraint G[124]!=G[54]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) maximize S; 
output ["G = \(G)\nObj=\(S)"]; 
