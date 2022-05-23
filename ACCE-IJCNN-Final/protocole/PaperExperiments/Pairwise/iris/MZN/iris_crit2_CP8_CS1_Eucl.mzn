include "globals.mzn"; 

int: n; % number of points 
int: k_min; 
int: k_max; 

array[1..n, 1..k_max] of int: M; 
array[1..n] of int: P; 
array[1..n] of var 1..k_max: G; 

var min(M) .. max(M) : Obj; 

constraint forall (i in 1..n, k in 1..k_max where P[i]=k) ( M[i,k] > Obj -> P[i]=G[i]); 


constraint G[143]!=G[12]; 
constraint G[33]=G[18]; 
constraint G[126]=G[121]; 
constraint G[41]!=G[131]; 
constraint G[36]=G[33]; 
constraint G[89]!=G[125]; 
constraint G[135]!=G[27]; 
constraint G[91]!=G[117]; 
constraint G[55]!=G[20]; 
constraint G[69]!=G[7]; 
constraint G[130]=G[104]; 
constraint G[80]=G[57]; 
constraint G[55]=G[75]; 
constraint G[120]!=G[61]; 
constraint G[60]!=G[36]; 
constraint G[148]=G[123]; 
constraint G[2]!=G[104]; 
constraint G[12]!=G[110]; 
constraint G[112]!=G[88]; 
constraint G[140]!=G[97]; 
constraint G[16]!=G[146]; 
constraint G[10]=G[30]; 
constraint G[120]=G[113]; 
constraint G[118]!=G[35]; 
constraint G[127]=G[133]; 
constraint G[142]!=G[9]; 
constraint G[144]=G[123]; 
constraint G[50]=G[12]; 
constraint G[62]=G[73]; 
constraint G[1]!=G[76]; 
constraint G[145]!=G[13]; 
constraint G[111]!=G[56]; 
constraint G[70]=G[61]; 
constraint G[77]!=G[148]; 
constraint G[81]=G[74]; 
constraint G[28]!=G[142]; 
constraint G[114]=G[139]; 
constraint G[25]=G[15]; 
constraint G[97]!=G[24]; 
constraint G[16]!=G[102]; 
constraint G[130]!=G[85]; 
constraint G[66]=G[76]; 
constraint G[93]=G[92]; 
constraint G[77]!=G[45]; 
constraint G[141]=G[112]; 
constraint G[45]!=G[85]; 
constraint G[78]!=G[110]; 
constraint G[141]!=G[47]; 
constraint G[28]!=G[128]; 
constraint G[66]!=G[32]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) minimize Obj; 
output ["G = \(G)\nObj=\(Obj)"]; 
