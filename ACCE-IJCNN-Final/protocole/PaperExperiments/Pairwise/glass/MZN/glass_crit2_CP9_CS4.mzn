include "globals.mzn"; 

int: n; % number of points 
int: k_min; 
int: k_max; 

array[1..n, 1..k_max] of int: M; 
array[1..n] of int: P; 
array[1..n] of var 1..k_max: G; 

var min(M) .. max(M) : Obj; 

constraint forall (i in 1..n, k in 1..k_max where P[i]=k) ( M[i,k] > Obj -> P[i]=G[i]); 


constraint G[65]!=G[131]; 
constraint G[121]!=G[161]; 
constraint G[200]!=G[146]; 
constraint G[27]=G[3]; 
constraint G[177]!=G[85]; 
constraint G[52]!=G[92]; 
constraint G[182]!=G[93]; 
constraint G[19]!=G[210]; 
constraint G[38]=G[55]; 
constraint G[105]=G[83]; 
constraint G[55]!=G[130]; 
constraint G[47]=G[4]; 
constraint G[98]=G[84]; 
constraint G[35]=G[7]; 
constraint G[114]!=G[206]; 
constraint G[111]=G[80]; 
constraint G[112]=G[110]; 
constraint G[128]!=G[9]; 
constraint G[93]!=G[60]; 
constraint G[63]=G[8]; 
constraint G[41]=G[28]; 
constraint G[177]!=G[115]; 
constraint G[202]!=G[152]; 
constraint G[22]=G[28]; 
constraint G[10]!=G[164]; 
constraint G[203]=G[202]; 
constraint G[141]=G[98]; 
constraint G[35]!=G[79]; 
constraint G[160]!=G[92]; 
constraint G[158]!=G[70]; 
constraint G[49]!=G[146]; 
constraint G[142]!=G[186]; 
constraint G[129]!=G[212]; 
constraint G[56]!=G[97]; 
constraint G[195]!=G[79]; 
constraint G[213]!=G[103]; 
constraint G[142]!=G[206]; 
constraint G[151]!=G[121]; 
constraint G[146]!=G[156]; 
constraint G[29]!=G[144]; 
constraint G[189]!=G[175]; 
constraint G[126]!=G[9]; 
constraint G[123]=G[146]; 
constraint G[155]!=G[128]; 
constraint G[21]=G[26]; 
constraint G[86]!=G[202]; 
constraint G[188]!=G[109]; 
constraint G[157]!=G[7]; 
constraint G[8]!=G[109]; 
constraint G[201]!=G[123]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) minimize Obj; 
output ["G = \(G)\nObj=\(Obj)"]; 
