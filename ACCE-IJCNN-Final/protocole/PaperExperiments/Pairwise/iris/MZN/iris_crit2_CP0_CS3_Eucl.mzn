include "globals.mzn"; 

int: n; % number of points 
int: k_min; 
int: k_max; 

array[1..n, 1..k_max] of int: M; 
array[1..n] of int: P; 
array[1..n] of var 1..k_max: G; 

var min(M) .. max(M) : Obj; 

constraint forall (i in 1..n, k in 1..k_max where P[i]=k) ( M[i,k] > Obj -> P[i]=G[i]); 


constraint G[60]!=G[34]; 
constraint G[91]=G[59]; 
constraint G[57]=G[89]; 
constraint G[78]!=G[46]; 
constraint G[3]!=G[76]; 
constraint G[59]!=G[146]; 
constraint G[23]!=G[95]; 
constraint G[58]!=G[118]; 
constraint G[45]!=G[132]; 
constraint G[21]=G[29]; 
constraint G[146]=G[131]; 
constraint G[29]=G[10]; 
constraint G[29]!=G[119]; 
constraint G[69]!=G[6]; 
constraint G[56]=G[60]; 
constraint G[133]!=G[58]; 
constraint G[26]=G[14]; 
constraint G[113]=G[148]; 
constraint G[50]!=G[106]; 
constraint G[12]=G[34]; 
constraint G[116]!=G[5]; 
constraint G[91]=G[97]; 
constraint G[120]=G[110]; 
constraint G[14]=G[48]; 
constraint G[84]=G[68]; 
constraint G[53]!=G[128]; 
constraint G[106]!=G[80]; 
constraint G[47]!=G[125]; 
constraint G[7]=G[25]; 
constraint G[102]!=G[31]; 
constraint G[48]!=G[51]; 
constraint G[69]!=G[107]; 
constraint G[46]=G[50]; 
constraint G[136]!=G[55]; 
constraint G[68]=G[80]; 
constraint G[63]!=G[24]; 
constraint G[122]=G[108]; 
constraint G[15]!=G[61]; 
constraint G[49]!=G[84]; 
constraint G[93]=G[74]; 
constraint G[113]=G[141]; 
constraint G[10]!=G[99]; 
constraint G[21]!=G[67]; 
constraint G[134]=G[117]; 
constraint G[43]=G[32]; 
constraint G[107]=G[139]; 
constraint G[122]!=G[70]; 
constraint G[71]!=G[120]; 
constraint G[90]!=G[43]; 
constraint G[130]=G[150]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) minimize Obj; 
output ["G = \(G)\nObj=\(Obj)"]; 
