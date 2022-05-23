include "globals.mzn"; 

int: n; % number of points 
int: k_min; 
int: k_max; 

array[1..n, 1..k_max] of int: M; 
array[1..n] of int: P; 
array[1..n] of var 1..k_max: G; 

var min(M) .. max(M) : Obj; 

constraint forall (i in 1..n, k in 1..k_max where P[i]=k) ( M[i,k] > Obj -> P[i]=G[i]); 


constraint G[97]=G[78]; 
constraint G[127]!=G[37]; 
constraint G[138]=G[140]; 
constraint G[114]=G[117]; 
constraint G[98]!=G[20]; 
constraint G[94]!=G[104]; 
constraint G[58]=G[99]; 
constraint G[109]!=G[57]; 
constraint G[77]!=G[15]; 
constraint G[45]!=G[137]; 
constraint G[48]!=G[79]; 
constraint G[115]!=G[82]; 
constraint G[114]=G[122]; 
constraint G[105]=G[111]; 
constraint G[72]!=G[39]; 
constraint G[33]!=G[88]; 
constraint G[124]!=G[59]; 
constraint G[137]!=G[51]; 
constraint G[112]=G[135]; 
constraint G[66]=G[62]; 
constraint G[17]!=G[106]; 
constraint G[64]!=G[145]; 
constraint G[39]!=G[146]; 
constraint G[142]!=G[63]; 
constraint G[21]!=G[88]; 
constraint G[35]!=G[65]; 
constraint G[63]=G[95]; 
constraint G[5]=G[8]; 
constraint G[39]!=G[53]; 
constraint G[116]!=G[97]; 
constraint G[136]=G[114]; 
constraint G[122]!=G[66]; 
constraint G[103]=G[145]; 
constraint G[147]=G[114]; 
constraint G[142]!=G[31]; 
constraint G[115]=G[117]; 
constraint G[69]!=G[2]; 
constraint G[82]=G[97]; 
constraint G[115]!=G[26]; 
constraint G[24]!=G[109]; 
constraint G[21]!=G[107]; 
constraint G[5]=G[31]; 
constraint G[38]=G[33]; 
constraint G[24]=G[13]; 
constraint G[140]!=G[50]; 
constraint G[62]!=G[44]; 
constraint G[102]!=G[98]; 
constraint G[61]!=G[36]; 
constraint G[121]!=G[47]; 
constraint G[62]!=G[34]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) minimize Obj; 
output ["G = \(G)\nObj=\(Obj)"]; 
