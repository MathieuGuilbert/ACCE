include "globals.mzn"; 

int: n; % number of points 
int: k_min; 
int: k_max; 

array[1..n, 1..k_max] of float: M; 
array[1..n] of int: P; 
array[1..n] of var 1..k_max: G; 

var min(i in 1..n, k in 1..k_max)(M[i,k]) .. max(i in 1..n, k in 1..k_max)(M[i,k]) : Obj; 

constraint forall (i in 1..n, k in 1..k_max where P[i]=k) ( M[i,k] > Obj -> P[i]=G[i]); 
constraint G[125]=G[101]; 
constraint G[17]!=G[93]; 
constraint G[63]!=G[146]; 
constraint G[70]=G[55]; 
constraint G[138]=G[101]; 
constraint G[21]!=G[73]; 
constraint G[76]!=G[139]; 
constraint G[142]!=G[9]; 
constraint G[137]!=G[98]; 
constraint G[19]!=G[110]; 
constraint G[147]!=G[13]; 
constraint G[15]!=G[70]; 
constraint G[67]!=G[21]; 
constraint G[63]!=G[40]; 
constraint G[59]=G[70]; 
constraint G[64]=G[96]; 
constraint G[84]=G[99]; 
constraint G[146]!=G[44]; 
constraint G[114]=G[113]; 
constraint G[67]=G[98]; 
constraint G[48]=G[44]; 
constraint G[134]!=G[7]; 
constraint G[46]=G[24]; 
constraint G[104]=G[123]; 
constraint G[124]!=G[12]; 
constraint G[135]!=G[10]; 
constraint G[128]=G[138]; 
constraint G[120]!=G[67]; 
constraint G[53]!=G[130]; 
constraint G[63]!=G[3]; 
constraint G[10]!=G[89]; 
constraint G[100]!=G[38]; 
constraint G[69]!=G[110]; 
constraint G[108]!=G[63]; 
constraint G[67]!=G[110]; 
constraint G[95]!=G[19]; 
constraint G[33]!=G[131]; 
constraint G[98]=G[83]; 
constraint G[25]=G[21]; 
constraint G[112]!=G[74]; 
constraint G[101]!=G[14]; 
constraint G[116]=G[148]; 
constraint G[82]!=G[113]; 
constraint G[29]=G[23]; 
constraint G[140]!=G[73]; 
constraint G[48]!=G[93]; 
constraint G[99]!=G[42]; 
constraint G[130]!=G[64]; 
constraint G[6]=G[12]; 
constraint G[94]=G[57]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) minimize Obj; 
output ["G = \(G)\nObj=\(Obj)"]; 
