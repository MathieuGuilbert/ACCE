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

constraint G[124]!=G[13]; 
constraint G[80]!=G[31]; 
constraint G[52]=G[97]; 
constraint G[94]!=G[27]; 
constraint G[135]!=G[23]; 
constraint G[2]!=G[87]; 
constraint G[35]!=G[58]; 
constraint G[147]!=G[48]; 
constraint G[108]!=G[61]; 
constraint G[9]!=G[131]; 
constraint G[62]!=G[33]; 
constraint G[40]!=G[122]; 
constraint G[105]!=G[85]; 
constraint G[27]!=G[135]; 
constraint G[56]=G[97]; 
constraint G[33]!=G[60]; 
constraint G[129]=G[150]; 
constraint G[9]!=G[116]; 
constraint G[20]!=G[69]; 
constraint G[99]=G[82]; 
constraint G[43]=G[41]; 
constraint G[140]!=G[97]; 
constraint G[38]!=G[82]; 
constraint G[28]=G[13]; 
constraint G[51]=G[88]; 
constraint G[147]=G[101]; 
constraint G[49]=G[42]; 
constraint G[83]=G[76]; 
constraint G[102]!=G[30]; 
constraint G[15]!=G[69]; 
constraint G[99]!=G[38]; 
constraint G[68]!=G[18]; 
constraint G[122]!=G[37]; 
constraint G[77]=G[60]; 
constraint G[87]=G[64]; 
constraint G[131]!=G[51]; 
constraint G[36]=G[19]; 
constraint G[93]=G[61]; 
constraint G[1]=G[24]; 
constraint G[58]=G[55]; 
constraint G[133]!=G[85]; 
constraint G[14]!=G[92]; 
constraint G[31]!=G[129]; 
constraint G[70]!=G[12]; 
constraint G[82]!=G[122]; 
constraint G[143]!=G[39]; 
constraint G[58]!=G[11]; 
constraint G[35]=G[24]; 
constraint G[89]!=G[24]; 
constraint G[5]!=G[119]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) maximize S; 
output ["G = \(G)\nObj=\(S)"]; 
