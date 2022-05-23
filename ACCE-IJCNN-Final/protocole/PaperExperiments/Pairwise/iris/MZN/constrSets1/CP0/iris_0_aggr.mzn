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

constraint G[13]!=G[119]; 
constraint G[79]!=G[132]; 
constraint G[60]!=G[126]; 
constraint G[32]!=G[72]; 
constraint G[116]!=G[16]; 
constraint G[56]!=G[112]; 
constraint G[24]!=G[56]; 
constraint G[68]!=G[101]; 
constraint G[1]!=G[114]; 
constraint G[17]!=G[110]; 
constraint G[79]!=G[23]; 
constraint G[16]!=G[125]; 
constraint G[128]!=G[56]; 
constraint G[54]!=G[25]; 
constraint G[83]!=G[22]; 
constraint G[105]!=G[78]; 
constraint G[130]!=G[8]; 
constraint G[98]!=G[40]; 
constraint G[94]!=G[39]; 
constraint G[109]!=G[69]; 
constraint G[110]!=G[66]; 
constraint G[81]!=G[103]; 
constraint G[56]!=G[96]; 
constraint G[11]!=G[90]; 
constraint G[48]!=G[126]; 
constraint G[124]!=G[45]; 
constraint G[70]!=G[136]; 
constraint G[100]!=G[49]; 
constraint G[109]!=G[75]; 
constraint G[67]!=G[116]; 
constraint G[96]!=G[56]; 
constraint G[14]!=G[105]; 
constraint G[99]!=G[2]; 
constraint G[33]!=G[95]; 
constraint G[5]!=G[62]; 
constraint G[3]!=G[124]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) maximize S; 
output ["G = \(G)\nObj=\(S)"]; 
