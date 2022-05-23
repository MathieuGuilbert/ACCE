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

constraint G[41]!=G[69]; 
constraint G[63]!=G[25]; 
constraint G[11]!=G[130]; 
constraint G[48]!=G[100]; 
constraint G[82]!=G[26]; 
constraint G[107]!=G[60]; 
constraint G[47]!=G[39]; 
constraint G[66]!=G[10]; 
constraint G[110]!=G[9]; 
constraint G[79]!=G[112]; 
constraint G[89]!=G[30]; 
constraint G[123]!=G[34]; 
constraint G[62]!=G[10]; 
constraint G[47]!=G[122]; 
constraint G[72]!=G[27]; 
constraint G[126]!=G[5]; 
constraint G[24]!=G[64]; 
constraint G[29]!=G[91]; 
constraint G[90]!=G[3]; 
constraint G[31]!=G[96]; 
constraint G[74]!=G[132]; 
constraint G[131]!=G[42]; 
constraint G[84]!=G[8]; 
constraint G[130]!=G[10]; 
constraint G[20]!=G[64]; 
constraint G[83]!=G[102]; 
constraint G[53]!=G[45]; 
constraint G[129]!=G[51]; 
constraint G[135]!=G[23]; 
constraint G[38]!=G[103]; 
constraint G[12]!=G[95]; 
constraint G[113]!=G[76]; 
constraint G[75]!=G[18]; 
constraint G[8]!=G[74]; 
constraint G[104]!=G[11]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) maximize S; 
output ["G = \(G)\nObj=\(S)"]; 
