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

constraint G[71]!=G[121]; 
constraint G[76]!=G[37]; 
constraint G[35]!=G[81]; 
constraint G[122]!=G[42]; 
constraint G[75]!=G[31]; 
constraint G[3]!=G[131]; 
constraint G[62]!=G[14]; 
constraint G[3]!=G[61]; 
constraint G[15]!=G[53]; 
constraint G[108]!=G[35]; 
constraint G[109]!=G[75]; 
constraint G[103]!=G[69]; 
constraint G[97]!=G[65]; 
constraint G[30]!=G[79]; 
constraint G[93]!=G[19]; 
constraint G[34]!=G[126]; 
constraint G[95]!=G[60]; 
constraint G[52]!=G[98]; 
constraint G[54]!=G[7]; 
constraint G[119]!=G[47]; 
constraint G[14]!=G[82]; 
constraint G[4]!=G[56]; 
constraint G[116]!=G[22]; 
constraint G[69]!=G[110]; 
constraint G[109]!=G[89]; 
constraint G[125]!=G[23]; 
constraint G[56]!=G[24]; 
constraint G[97]!=G[6]; 
constraint G[75]!=G[46]; 
constraint G[132]!=G[1]; 
constraint G[21]!=G[86]; 
constraint G[15]!=G[95]; 
constraint G[52]!=G[25]; 
constraint G[2]!=G[106]; 
constraint G[127]!=G[10]; 
constraint G[80]!=G[95]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) maximize S; 
output ["G = \(G)\nObj=\(S)"]; 
