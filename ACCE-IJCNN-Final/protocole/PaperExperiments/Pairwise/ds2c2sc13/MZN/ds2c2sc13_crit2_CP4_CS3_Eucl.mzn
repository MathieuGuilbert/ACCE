include "globals.mzn"; 

int: n; % number of points 
int: k_min; 
int: k_max; 

array[1..n, 1..k_max] of int: M; 
array[1..n] of int: P; 
array[1..n] of var 1..k_max: G; 

var min(M) .. max(M) : Obj; 

constraint forall (i in 1..n, k in 1..k_max where P[i]=k) ( M[i,k] > Obj -> P[i]=G[i]); 


constraint G[416]!=G[210]; 
constraint G[120]!=G[510]; 
constraint G[3]!=G[300]; 
constraint G[3]!=G[224]; 
constraint G[414]!=G[428]; 
constraint G[564]!=G[333]; 
constraint G[450]!=G[96]; 
constraint G[379]!=G[179]; 
constraint G[251]!=G[567]; 
constraint G[88]!=G[169]; 
constraint G[235]!=G[474]; 
constraint G[43]!=G[469]; 
constraint G[504]!=G[89]; 
constraint G[201]!=G[419]; 
constraint G[207]!=G[290]; 
constraint G[295]!=G[575]; 
constraint G[269]!=G[113]; 
constraint G[34]!=G[203]; 
constraint G[88]!=G[176]; 
constraint G[367]!=G[278]; 
constraint G[539]!=G[25]; 
constraint G[456]!=G[229]; 
constraint G[228]!=G[356]; 
constraint G[346]!=G[458]; 
constraint G[502]!=G[260]; 
constraint G[69]!=G[348]; 
constraint G[162]!=G[17]; 
constraint G[374]!=G[500]; 
constraint G[424]!=G[322]; 
constraint G[405]=G[400]; 
constraint G[250]!=G[265]; 
constraint G[422]!=G[84]; 
constraint G[573]!=G[254]; 
constraint G[283]!=G[412]; 
constraint G[100]!=G[382]; 
constraint G[517]!=G[422]; 
constraint G[157]!=G[228]; 
constraint G[488]!=G[364]; 
constraint G[102]!=G[581]; 
constraint G[312]!=G[153]; 
constraint G[483]=G[479]; 
constraint G[97]!=G[418]; 
constraint G[4]!=G[237]; 
constraint G[551]!=G[245]; 
constraint G[245]!=G[395]; 
constraint G[477]=G[489]; 
constraint G[306]!=G[24]; 
constraint G[315]!=G[168]; 
constraint G[142]!=G[437]; 
constraint G[201]!=G[325]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) minimize Obj; 
output ["G = \(G)\nObj=\(Obj)"]; 
