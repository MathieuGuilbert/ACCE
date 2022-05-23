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

constraint G[38]!=G[45]; 
constraint G[59]!=G[7]; 
constraint G[92]!=G[31]; 
constraint G[89]!=G[58]; 
constraint G[42]!=G[87]; 
constraint G[20]!=G[71]; 
constraint G[16]!=G[104]; 
constraint G[71]!=G[111]; 
constraint G[101]!=G[44]; 
constraint G[94]!=G[72]; 
constraint G[113]!=G[35]; 
constraint G[78]!=G[21]; 
constraint G[33]!=G[113]; 
constraint G[85]!=G[11]; 
constraint G[111]!=G[18]; 
constraint G[10]!=G[56]; 
constraint G[129]!=G[44]; 
constraint G[111]!=G[60]; 
constraint G[94]!=G[5]; 
constraint G[132]!=G[56]; 
constraint G[113]!=G[83]; 
constraint G[2]!=G[82]; 
constraint G[124]!=G[49]; 
constraint G[79]!=G[132]; 
constraint G[114]!=G[38]; 
constraint G[127]!=G[3]; 
constraint G[97]!=G[30]; 
constraint G[11]!=G[62]; 
constraint G[118]!=G[85]; 
constraint G[57]!=G[125]; 
constraint G[82]!=G[98]; 
constraint G[71]!=G[91]; 
constraint G[35]!=G[89]; 
constraint G[57]!=G[88]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) maximize S; 
output ["G = \(G)\nObj=\(S)"]; 
