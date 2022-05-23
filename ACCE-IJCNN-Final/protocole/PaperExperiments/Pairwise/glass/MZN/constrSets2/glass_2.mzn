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

constraint G[211]=G[214]; 
constraint G[81]!=G[54]; 
constraint G[103]=G[110]; 
constraint G[85]=G[72]; 
constraint G[60]!=G[103]; 
constraint G[120]!=G[4]; 
constraint G[144]=G[127]; 
constraint G[79]!=G[189]; 
constraint G[200]!=G[122]; 
constraint G[185]!=G[73]; 
constraint G[30]=G[40]; 
constraint G[182]!=G[80]; 
constraint G[121]!=G[57]; 
constraint G[91]=G[128]; 
constraint G[185]!=G[197]; 
constraint G[55]!=G[181]; 
constraint G[180]!=G[27]; 
constraint G[114]=G[84]; 
constraint G[145]!=G[8]; 
constraint G[205]!=G[98]; 
constraint G[86]=G[98]; 
constraint G[199]!=G[72]; 
constraint G[158]!=G[183]; 
constraint G[212]!=G[126]; 
constraint G[166]!=G[10]; 
constraint G[151]=G[156]; 
constraint G[181]!=G[36]; 
constraint G[68]!=G[75]; 
constraint G[200]!=G[165]; 
constraint G[122]!=G[188]; 
constraint G[4]!=G[94]; 
constraint G[9]!=G[124]; 
constraint G[72]!=G[163]; 
constraint G[160]!=G[23]; 
constraint G[193]!=G[139]; 
constraint G[209]!=G[171]; 
constraint G[107]=G[115]; 
constraint G[29]=G[31]; 
constraint G[179]!=G[66]; 
constraint G[113]!=G[5]; 
constraint G[16]!=G[202]; 
constraint G[14]!=G[119]; 
constraint G[15]!=G[139]; 
constraint G[193]!=G[86]; 
constraint G[40]!=G[204]; 
constraint G[132]!=G[6]; 
constraint G[129]!=G[26]; 
constraint G[150]!=G[213]; 
constraint G[107]=G[78]; 
constraint G[132]=G[117]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) maximize S; 
output ["G = \(G)\nObj=\(S)"]; 
