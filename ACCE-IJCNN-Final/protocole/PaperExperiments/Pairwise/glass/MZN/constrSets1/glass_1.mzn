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

constraint G[5]=G[2]; 
constraint G[194]!=G[141]; 
constraint G[201]=G[214]; 
constraint G[48]!=G[90]; 
constraint G[70]=G[23]; 
constraint G[101]!=G[20]; 
constraint G[176]!=G[99]; 
constraint G[139]!=G[148]; 
constraint G[19]!=G[93]; 
constraint G[67]!=G[80]; 
constraint G[25]!=G[102]; 
constraint G[128]!=G[194]; 
constraint G[140]=G[95]; 
constraint G[102]=G[82]; 
constraint G[36]!=G[148]; 
constraint G[11]!=G[95]; 
constraint G[17]!=G[151]; 
constraint G[144]=G[131]; 
constraint G[1]!=G[173]; 
constraint G[20]!=G[178]; 
constraint G[78]!=G[64]; 
constraint G[119]=G[141]; 
constraint G[117]!=G[166]; 
constraint G[67]!=G[114]; 
constraint G[78]!=G[158]; 
constraint G[136]=G[83]; 
constraint G[142]=G[146]; 
constraint G[81]=G[85]; 
constraint G[112]!=G[49]; 
constraint G[162]!=G[41]; 
constraint G[40]=G[39]; 
constraint G[42]!=G[144]; 
constraint G[10]=G[67]; 
constraint G[153]!=G[5]; 
constraint G[37]!=G[157]; 
constraint G[200]!=G[3]; 
constraint G[8]!=G[163]; 
constraint G[156]!=G[141]; 
constraint G[129]!=G[6]; 
constraint G[85]!=G[1]; 
constraint G[92]!=G[157]; 
constraint G[36]!=G[149]; 
constraint G[189]=G[187]; 
constraint G[151]!=G[65]; 
constraint G[22]=G[43]; 
constraint G[21]!=G[199]; 
constraint G[55]!=G[199]; 
constraint G[123]!=G[61]; 
constraint G[73]!=G[186]; 
constraint G[213]!=G[130]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) maximize S; 
output ["G = \(G)\nObj=\(S)"]; 
