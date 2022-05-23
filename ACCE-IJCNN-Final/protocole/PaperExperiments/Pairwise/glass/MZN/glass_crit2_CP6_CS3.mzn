include "globals.mzn"; 

int: n; % number of points 
int: k_min; 
int: k_max; 

array[1..n, 1..k_max] of int: M; 
array[1..n] of int: P; 
array[1..n] of var 1..k_max: G; 

var min(M) .. max(M) : Obj; 

constraint forall (i in 1..n, k in 1..k_max where P[i]=k) ( M[i,k] > Obj -> P[i]=G[i]); 


constraint G[193]!=G[71]; 
constraint G[38]!=G[187]; 
constraint G[3]!=G[149]; 
constraint G[131]!=G[148]; 
constraint G[143]!=G[197]; 
constraint G[4]=G[56]; 
constraint G[125]!=G[55]; 
constraint G[29]!=G[79]; 
constraint G[84]!=G[168]; 
constraint G[146]!=G[40]; 
constraint G[163]!=G[143]; 
constraint G[152]!=G[85]; 
constraint G[196]=G[213]; 
constraint G[4]!=G[157]; 
constraint G[74]!=G[63]; 
constraint G[15]=G[62]; 
constraint G[34]!=G[78]; 
constraint G[175]!=G[160]; 
constraint G[141]!=G[30]; 
constraint G[109]=G[107]; 
constraint G[136]!=G[167]; 
constraint G[126]=G[137]; 
constraint G[16]!=G[129]; 
constraint G[40]=G[52]; 
constraint G[188]!=G[108]; 
constraint G[135]=G[90]; 
constraint G[193]!=G[5]; 
constraint G[24]=G[8]; 
constraint G[204]!=G[2]; 
constraint G[59]!=G[186]; 
constraint G[113]!=G[202]; 
constraint G[204]!=G[83]; 
constraint G[86]=G[80]; 
constraint G[187]!=G[89]; 
constraint G[45]!=G[179]; 
constraint G[76]=G[145]; 
constraint G[125]=G[84]; 
constraint G[174]!=G[122]; 
constraint G[52]!=G[197]; 
constraint G[63]!=G[200]; 
constraint G[67]!=G[202]; 
constraint G[51]=G[57]; 
constraint G[123]!=G[2]; 
constraint G[116]=G[75]; 
constraint G[175]!=G[43]; 
constraint G[72]!=G[44]; 
constraint G[120]!=G[212]; 
constraint G[37]!=G[114]; 
constraint G[60]!=G[77]; 
constraint G[41]!=G[89]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) minimize Obj; 
output ["G = \(G)\nObj=\(Obj)"]; 
