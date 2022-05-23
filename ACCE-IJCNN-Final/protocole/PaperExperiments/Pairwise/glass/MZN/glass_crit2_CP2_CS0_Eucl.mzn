include "globals.mzn"; 

int: n; % number of points 
int: k_min; 
int: k_max; 

array[1..n, 1..k_max] of int: M; 
array[1..n] of int: P; 
array[1..n] of var 1..k_max: G; 

var min(M) .. max(M) : Obj; 

constraint forall (i in 1..n, k in 1..k_max where P[i]=k) ( M[i,k] > Obj -> P[i]=G[i]); 


constraint G[117]!=G[31]; 
constraint G[92]!=G[67]; 
constraint G[49]!=G[208]; 
constraint G[132]!=G[1]; 
constraint G[77]!=G[17]; 
constraint G[42]!=G[176]; 
constraint G[169]=G[176]; 
constraint G[143]!=G[52]; 
constraint G[28]!=G[214]; 
constraint G[172]!=G[78]; 
constraint G[103]=G[107]; 
constraint G[156]!=G[119]; 
constraint G[177]!=G[76]; 
constraint G[154]!=G[22]; 
constraint G[188]!=G[88]; 
constraint G[89]=G[127]; 
constraint G[29]!=G[151]; 
constraint G[154]=G[152]; 
constraint G[203]!=G[62]; 
constraint G[85]!=G[15]; 
constraint G[36]=G[13]; 
constraint G[163]!=G[205]; 
constraint G[56]!=G[84]; 
constraint G[119]!=G[155]; 
constraint G[53]=G[1]; 
constraint G[57]!=G[152]; 
constraint G[171]!=G[126]; 
constraint G[94]=G[121]; 
constraint G[104]=G[117]; 
constraint G[114]!=G[170]; 
constraint G[177]!=G[88]; 
constraint G[58]!=G[197]; 
constraint G[7]=G[32]; 
constraint G[147]!=G[129]; 
constraint G[5]!=G[121]; 
constraint G[61]=G[48]; 
constraint G[5]!=G[80]; 
constraint G[62]!=G[159]; 
constraint G[107]=G[74]; 
constraint G[51]!=G[186]; 
constraint G[98]!=G[30]; 
constraint G[117]!=G[30]; 
constraint G[9]!=G[99]; 
constraint G[56]!=G[201]; 
constraint G[128]!=G[184]; 
constraint G[53]!=G[147]; 
constraint G[82]=G[73]; 
constraint G[110]=G[133]; 
constraint G[46]!=G[169]; 
constraint G[24]!=G[140]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) minimize Obj; 
output ["G = \(G)\nObj=\(Obj)"]; 
