include "globals.mzn"; 

int: n; % number of points 
int: k_min; 
int: k_max; 

array[1..n, 1..k_max] of int: M; 
array[1..n] of int: P; 
array[1..n] of var 1..k_max: G; 

var min(M) .. max(M) : Obj; 

constraint forall (i in 1..n, k in 1..k_max where P[i]=k) ( M[i,k] > Obj -> P[i]=G[i]); 


constraint G[12]!=G[75]; 
constraint G[63]=G[83]; 
constraint G[98]!=G[33]; 
constraint G[122]!=G[29]; 
constraint G[128]=G[119]; 
constraint G[116]!=G[81]; 
constraint G[88]=G[96]; 
constraint G[109]=G[122]; 
constraint G[121]!=G[54]; 
constraint G[93]!=G[41]; 
constraint G[50]!=G[97]; 
constraint G[127]!=G[17]; 
constraint G[48]!=G[146]; 
constraint G[5]!=G[144]; 
constraint G[104]!=G[78]; 
constraint G[117]!=G[56]; 
constraint G[130]!=G[51]; 
constraint G[46]!=G[120]; 
constraint G[113]!=G[84]; 
constraint G[8]!=G[60]; 
constraint G[62]=G[75]; 
constraint G[113]!=G[98]; 
constraint G[39]!=G[88]; 
constraint G[77]!=G[119]; 
constraint G[29]!=G[101]; 
constraint G[64]!=G[27]; 
constraint G[75]=G[88]; 
constraint G[20]!=G[73]; 
constraint G[40]!=G[63]; 
constraint G[47]!=G[139]; 
constraint G[113]!=G[85]; 
constraint G[131]!=G[58]; 
constraint G[138]!=G[12]; 
constraint G[10]!=G[91]; 
constraint G[3]!=G[52]; 
constraint G[99]=G[62]; 
constraint G[84]!=G[38]; 
constraint G[111]=G[131]; 
constraint G[56]!=G[42]; 
constraint G[8]!=G[58]; 
constraint G[41]!=G[75]; 
constraint G[43]!=G[66]; 
constraint G[98]!=G[112]; 
constraint G[83]=G[52]; 
constraint G[10]=G[31]; 
constraint G[1]=G[48]; 
constraint G[128]!=G[78]; 
constraint G[134]=G[123]; 
constraint G[64]!=G[121]; 
constraint G[113]=G[132]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) minimize Obj; 
output ["G = \(G)\nObj=\(Obj)"]; 
