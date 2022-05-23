include "globals.mzn"; 

int: n; % number of points 
int: k_min; 
int: k_max; 

array[1..n, 1..k_max] of int: M; 
array[1..n] of int: P; 
array[1..n] of var 1..k_max: G; 

var min(M) .. max(M) : Obj; 

constraint forall (i in 1..n, k in 1..k_max where P[i]=k) ( M[i,k] > Obj -> P[i]=G[i]); 


constraint G[54]=G[65]; 
constraint G[38]=G[14]; 
constraint G[7]!=G[101]; 
constraint G[73]!=G[130]; 
constraint G[138]!=G[33]; 
constraint G[62]=G[68]; 
constraint G[27]=G[46]; 
constraint G[32]=G[16]; 
constraint G[42]!=G[135]; 
constraint G[135]!=G[82]; 
constraint G[77]!=G[129]; 
constraint G[70]!=G[104]; 
constraint G[108]!=G[19]; 
constraint G[70]=G[57]; 
constraint G[71]!=G[19]; 
constraint G[43]=G[19]; 
constraint G[125]=G[110]; 
constraint G[105]=G[126]; 
constraint G[8]=G[7]; 
constraint G[45]!=G[66]; 
constraint G[30]!=G[146]; 
constraint G[10]!=G[62]; 
constraint G[52]=G[78]; 
constraint G[4]!=G[89]; 
constraint G[38]!=G[57]; 
constraint G[19]!=G[118]; 
constraint G[144]!=G[38]; 
constraint G[7]!=G[53]; 
constraint G[124]!=G[4]; 
constraint G[6]=G[12]; 
constraint G[116]!=G[87]; 
constraint G[78]=G[97]; 
constraint G[96]=G[63]; 
constraint G[114]=G[134]; 
constraint G[50]!=G[60]; 
constraint G[33]=G[4]; 
constraint G[86]!=G[32]; 
constraint G[61]=G[90]; 
constraint G[2]!=G[145]; 
constraint G[105]!=G[47]; 
constraint G[36]!=G[107]; 
constraint G[99]!=G[47]; 
constraint G[66]=G[92]; 
constraint G[123]!=G[18]; 
constraint G[10]!=G[71]; 
constraint G[104]!=G[86]; 
constraint G[42]!=G[109]; 
constraint G[68]!=G[120]; 
constraint G[116]!=G[36]; 
constraint G[144]!=G[5]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) minimize Obj; 
output ["G = \(G)\nObj=\(Obj)"]; 
