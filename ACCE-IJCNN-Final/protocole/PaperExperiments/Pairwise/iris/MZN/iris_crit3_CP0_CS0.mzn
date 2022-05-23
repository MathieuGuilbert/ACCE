include "globals.mzn"; 

int: n; % number of points 
int: k_min; 
int: k_max; 

array[1..n, 1..k_max] of int: M; 
array[1..n] of int: P; 
array[1..n] of var 1..k_max: G; 

var min(M) .. max(M) : Obj; 

constraint forall (i in 1..n, k in 1..k_max where G[i]=k) ( M[i,k] < Obj -> P[i]=G[i]); 

constraint G[94]=G[69]; 
constraint G[7]!=G[77]; 
constraint G[53]!=G[107]; 
constraint G[1]!=G[58]; 
constraint G[65]!=G[108]; 
constraint G[47]=G[19]; 
constraint G[25]!=G[113]; 
constraint G[103]!=G[16]; 
constraint G[6]!=G[68]; 
constraint G[131]!=G[64]; 
constraint G[64]!=G[13]; 
constraint G[11]=G[38]; 
constraint G[68]!=G[104]; 
constraint G[59]=G[61]; 
constraint G[71]!=G[147]; 
constraint G[80]!=G[150]; 
constraint G[28]!=G[93]; 
constraint G[119]=G[126]; 
constraint G[18]=G[44]; 
constraint G[144]!=G[12]; 
constraint G[51]!=G[111]; 
constraint G[32]=G[8]; 
constraint G[112]!=G[46]; 
constraint G[126]!=G[100]; 
constraint G[122]!=G[60]; 
constraint G[66]=G[94]; 
constraint G[39]=G[4]; 
constraint G[45]!=G[68]; 
constraint G[46]!=G[80]; 
constraint G[16]!=G[109]; 
constraint G[1]!=G[125]; 
constraint G[138]=G[105]; 
constraint G[117]=G[129]; 
constraint G[116]=G[149]; 
constraint G[2]!=G[117]; 
constraint G[90]=G[61]; 
constraint G[29]!=G[72]; 
constraint G[56]!=G[104]; 
constraint G[19]=G[43]; 
constraint G[90]!=G[41]; 
constraint G[116]!=G[32]; 
constraint G[18]!=G[94]; 
constraint G[13]!=G[59]; 
constraint G[65]!=G[49]; 
constraint G[40]=G[20]; 
constraint G[16]!=G[105]; 
constraint G[92]!=G[145]; 
constraint G[107]!=G[91]; 
constraint G[44]=G[5]; 
constraint G[8]!=G[80]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_max) maximize Obj; 
output ["G = \(G)\nObj=\(Obj)"]; 
