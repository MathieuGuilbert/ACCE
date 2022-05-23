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

constraint G[81]!=G[27]; 
constraint G[201]!=G[172]; 
constraint G[193]=G[209]; 
constraint G[40]=G[25]; 
constraint G[163]!=G[69]; 
constraint G[20]!=G[165]; 
constraint G[24]!=G[121]; 
constraint G[167]!=G[190]; 
constraint G[152]=G[147]; 
constraint G[122]=G[73]; 
constraint G[181]!=G[70]; 
constraint G[197]!=G[116]; 
constraint G[110]!=G[157]; 
constraint G[60]!=G[84]; 
constraint G[76]!=G[173]; 
constraint G[165]=G[171]; 
constraint G[184]!=G[64]; 
constraint G[119]!=G[66]; 
constraint G[72]=G[111]; 
constraint G[97]!=G[177]; 
constraint G[49]=G[35]; 
constraint G[147]!=G[94]; 
constraint G[209]=G[194]; 
constraint G[175]!=G[136]; 
constraint G[53]!=G[197]; 
constraint G[211]=G[196]; 
constraint G[25]=G[61]; 
constraint G[207]!=G[44]; 
constraint G[102]!=G[55]; 
constraint G[84]!=G[213]; 
constraint G[2]!=G[153]; 
constraint G[38]!=G[158]; 
constraint G[56]!=G[199]; 
constraint G[106]!=G[171]; 
constraint G[66]!=G[176]; 
constraint G[32]=G[9]; 
constraint G[143]!=G[214]; 
constraint G[7]!=G[103]; 
constraint G[148]!=G[110]; 
constraint G[28]=G[42]; 
constraint G[40]=G[58]; 
constraint G[192]!=G[142]; 
constraint G[191]!=G[180]; 
constraint G[138]=G[121]; 
constraint G[97]=G[128]; 
constraint G[60]!=G[152]; 
constraint G[100]!=G[200]; 
constraint G[96]=G[86]; 
constraint G[211]!=G[48]; 
constraint G[194]!=G[49]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) maximize S; 
output ["G = \(G)\nObj=\(S)"]; 
