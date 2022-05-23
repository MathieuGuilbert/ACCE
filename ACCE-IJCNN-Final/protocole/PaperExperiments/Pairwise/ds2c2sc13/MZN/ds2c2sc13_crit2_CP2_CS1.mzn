include "globals.mzn"; 

int: n; % number of points 
int: k_min; 
int: k_max; 

array[1..n, 1..k_max] of int: M; 
array[1..n] of int: P; 
array[1..n] of var 1..k_max: G; 

var min(M) .. max(M) : Obj; 

constraint forall (i in 1..n, k in 1..k_max where P[i]=k) ( M[i,k] > Obj -> P[i]=G[i]); 


constraint G[14]!=G[387]; 
constraint G[153]!=G[529]; 
constraint G[276]!=G[325]; 
constraint G[514]!=G[131]; 
constraint G[370]!=G[108]; 
constraint G[212]!=G[302]; 
constraint G[89]!=G[417]; 
constraint G[91]!=G[580]; 
constraint G[469]!=G[291]; 
constraint G[568]!=G[165]; 
constraint G[344]!=G[90]; 
constraint G[456]!=G[354]; 
constraint G[574]!=G[484]; 
constraint G[404]!=G[249]; 
constraint G[49]!=G[241]; 
constraint G[202]!=G[446]; 
constraint G[244]!=G[494]; 
constraint G[506]!=G[266]; 
constraint G[432]!=G[27]; 
constraint G[43]!=G[167]; 
constraint G[511]!=G[425]; 
constraint G[285]!=G[194]; 
constraint G[35]!=G[420]; 
constraint G[520]!=G[185]; 
constraint G[475]!=G[156]; 
constraint G[531]!=G[18]; 
constraint G[135]!=G[309]; 
constraint G[572]!=G[51]; 
constraint G[297]!=G[374]; 
constraint G[564]!=G[100]; 
constraint G[78]!=G[308]; 
constraint G[492]!=G[19]; 
constraint G[408]!=G[418]; 
constraint G[306]=G[275]; 
constraint G[8]!=G[322]; 
constraint G[144]!=G[482]; 
constraint G[99]!=G[404]; 
constraint G[281]!=G[358]; 
constraint G[38]!=G[410]; 
constraint G[346]!=G[68]; 
constraint G[185]!=G[25]; 
constraint G[322]!=G[518]; 
constraint G[157]!=G[54]; 
constraint G[489]!=G[267]; 
constraint G[209]!=G[258]; 
constraint G[203]!=G[49]; 
constraint G[334]!=G[254]; 
constraint G[521]=G[524]; 
constraint G[489]=G[500]; 
constraint G[231]!=G[346]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) minimize Obj; 
output ["G = \(G)\nObj=\(Obj)"]; 
