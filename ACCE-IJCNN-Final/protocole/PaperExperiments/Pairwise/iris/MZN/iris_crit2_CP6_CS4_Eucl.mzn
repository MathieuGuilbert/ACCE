include "globals.mzn"; 

int: n; % number of points 
int: k_min; 
int: k_max; 

array[1..n, 1..k_max] of int: M; 
array[1..n] of int: P; 
array[1..n] of var 1..k_max: G; 

var min(M) .. max(M) : Obj; 

constraint forall (i in 1..n, k in 1..k_max where P[i]=k) ( M[i,k] > Obj -> P[i]=G[i]); 


constraint G[82]!=G[31]; 
constraint G[124]!=G[79]; 
constraint G[95]=G[66]; 
constraint G[14]=G[43]; 
constraint G[29]!=G[84]; 
constraint G[33]!=G[91]; 
constraint G[92]!=G[105]; 
constraint G[111]!=G[42]; 
constraint G[99]!=G[13]; 
constraint G[131]!=G[92]; 
constraint G[67]!=G[26]; 
constraint G[76]!=G[123]; 
constraint G[69]!=G[49]; 
constraint G[54]=G[99]; 
constraint G[114]!=G[98]; 
constraint G[37]=G[49]; 
constraint G[106]!=G[13]; 
constraint G[144]=G[129]; 
constraint G[34]!=G[90]; 
constraint G[47]=G[50]; 
constraint G[71]!=G[101]; 
constraint G[5]=G[19]; 
constraint G[50]!=G[143]; 
constraint G[86]!=G[26]; 
constraint G[134]!=G[98]; 
constraint G[134]=G[145]; 
constraint G[116]!=G[95]; 
constraint G[90]!=G[24]; 
constraint G[9]!=G[102]; 
constraint G[67]!=G[27]; 
constraint G[83]=G[97]; 
constraint G[90]=G[56]; 
constraint G[148]!=G[35]; 
constraint G[88]=G[61]; 
constraint G[127]=G[129]; 
constraint G[68]=G[65]; 
constraint G[20]!=G[110]; 
constraint G[30]=G[45]; 
constraint G[131]!=G[69]; 
constraint G[121]!=G[32]; 
constraint G[87]!=G[108]; 
constraint G[41]=G[33]; 
constraint G[28]=G[25]; 
constraint G[88]!=G[120]; 
constraint G[88]=G[86]; 
constraint G[121]!=G[94]; 
constraint G[25]=G[15]; 
constraint G[24]!=G[101]; 
constraint G[117]!=G[8]; 
constraint G[147]=G[137]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) minimize Obj; 
output ["G = \(G)\nObj=\(Obj)"]; 
