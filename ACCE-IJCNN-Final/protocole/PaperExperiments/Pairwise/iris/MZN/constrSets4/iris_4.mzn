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

constraint G[107]=G[139]; 
constraint G[13]!=G[101]; 
constraint G[41]=G[7]; 
constraint G[50]!=G[126]; 
constraint G[9]=G[28]; 
constraint G[106]=G[117]; 
constraint G[44]!=G[140]; 
constraint G[112]!=G[27]; 
constraint G[75]!=G[48]; 
constraint G[45]!=G[67]; 
constraint G[67]!=G[27]; 
constraint G[13]=G[50]; 
constraint G[132]!=G[36]; 
constraint G[69]!=G[49]; 
constraint G[12]!=G[97]; 
constraint G[46]!=G[130]; 
constraint G[121]=G[123]; 
constraint G[34]=G[24]; 
constraint G[64]=G[88]; 
constraint G[119]!=G[95]; 
constraint G[141]!=G[3]; 
constraint G[138]=G[120]; 
constraint G[92]!=G[139]; 
constraint G[136]!=G[6]; 
constraint G[126]=G[133]; 
constraint G[9]!=G[91]; 
constraint G[123]=G[127]; 
constraint G[45]=G[26]; 
constraint G[124]=G[146]; 
constraint G[37]!=G[143]; 
constraint G[58]=G[56]; 
constraint G[26]!=G[56]; 
constraint G[73]!=G[50]; 
constraint G[61]=G[52]; 
constraint G[117]!=G[15]; 
constraint G[145]!=G[82]; 
constraint G[133]!=G[58]; 
constraint G[54]!=G[132]; 
constraint G[134]!=G[65]; 
constraint G[85]!=G[121]; 
constraint G[99]!=G[37]; 
constraint G[97]!=G[13]; 
constraint G[44]!=G[80]; 
constraint G[82]!=G[117]; 
constraint G[59]=G[69]; 
constraint G[97]!=G[104]; 
constraint G[50]=G[25]; 
constraint G[116]!=G[80]; 
constraint G[144]=G[128]; 
constraint G[46]!=G[140]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) maximize S; 
output ["G = \(G)\nObj=\(S)"]; 
