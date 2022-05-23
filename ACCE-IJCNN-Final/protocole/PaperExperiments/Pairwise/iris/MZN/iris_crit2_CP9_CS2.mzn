include "globals.mzn"; 

int: n; % number of points 
int: k_min; 
int: k_max; 

array[1..n, 1..k_max] of int: M; 
array[1..n] of int: P; 
array[1..n] of var 1..k_max: G; 

var min(M) .. max(M) : Obj; 

constraint forall (i in 1..n, k in 1..k_max where P[i]=k) ( M[i,k] > Obj -> P[i]=G[i]); 


constraint G[137]=G[131]; 
constraint G[71]!=G[129]; 
constraint G[113]=G[119]; 
constraint G[118]=G[127]; 
constraint G[118]=G[119]; 
constraint G[28]!=G[51]; 
constraint G[138]=G[107]; 
constraint G[30]!=G[121]; 
constraint G[136]=G[102]; 
constraint G[131]=G[130]; 
constraint G[103]!=G[24]; 
constraint G[124]!=G[24]; 
constraint G[37]!=G[55]; 
constraint G[61]=G[74]; 
constraint G[131]!=G[50]; 
constraint G[150]!=G[77]; 
constraint G[41]!=G[57]; 
constraint G[129]=G[121]; 
constraint G[77]!=G[27]; 
constraint G[86]!=G[108]; 
constraint G[60]=G[53]; 
constraint G[1]!=G[91]; 
constraint G[12]=G[26]; 
constraint G[127]!=G[24]; 
constraint G[51]=G[78]; 
constraint G[95]=G[65]; 
constraint G[97]!=G[122]; 
constraint G[17]!=G[109]; 
constraint G[4]!=G[100]; 
constraint G[11]!=G[51]; 
constraint G[44]=G[41]; 
constraint G[66]!=G[42]; 
constraint G[1]=G[26]; 
constraint G[63]!=G[31]; 
constraint G[127]=G[111]; 
constraint G[133]=G[115]; 
constraint G[68]!=G[141]; 
constraint G[106]!=G[97]; 
constraint G[85]=G[96]; 
constraint G[30]!=G[105]; 
constraint G[110]=G[112]; 
constraint G[62]!=G[106]; 
constraint G[59]!=G[34]; 
constraint G[49]!=G[127]; 
constraint G[74]!=G[134]; 
constraint G[131]!=G[34]; 
constraint G[1]!=G[52]; 
constraint G[43]!=G[59]; 
constraint G[59]!=G[118]; 
constraint G[143]=G[130]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) minimize Obj; 
output ["G = \(G)\nObj=\(Obj)"]; 
