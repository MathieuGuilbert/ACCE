include "globals.mzn"; 

int: n; % number of points 
int: k_min; 
int: k_max; 

array[1..n, 1..k_max] of int: M; 
array[1..n] of int: P; 
array[1..n] of var 1..k_max: G; 

var min(M) .. max(M) : Obj; 

constraint forall (i in 1..n, k in 1..k_max where P[i]=k) ( M[i,k] > Obj -> P[i]=G[i]); 


constraint G[37]!=G[139]; 
constraint G[30]=G[25]; 
constraint G[22]!=G[64]; 
constraint G[39]!=G[114]; 
constraint G[143]!=G[81]; 
constraint G[45]!=G[141]; 
constraint G[96]!=G[140]; 
constraint G[100]=G[97]; 
constraint G[85]!=G[19]; 
constraint G[95]!=G[139]; 
constraint G[10]=G[41]; 
constraint G[25]=G[33]; 
constraint G[120]=G[108]; 
constraint G[104]!=G[25]; 
constraint G[56]=G[86]; 
constraint G[60]!=G[148]; 
constraint G[99]!=G[5]; 
constraint G[1]!=G[113]; 
constraint G[77]=G[83]; 
constraint G[82]!=G[124]; 
constraint G[139]!=G[86]; 
constraint G[23]!=G[93]; 
constraint G[3]=G[33]; 
constraint G[116]!=G[37]; 
constraint G[87]!=G[34]; 
constraint G[115]!=G[22]; 
constraint G[131]!=G[39]; 
constraint G[85]!=G[17]; 
constraint G[62]!=G[142]; 
constraint G[111]=G[119]; 
constraint G[38]=G[29]; 
constraint G[133]!=G[44]; 
constraint G[62]!=G[119]; 
constraint G[133]=G[113]; 
constraint G[86]=G[76]; 
constraint G[119]!=G[96]; 
constraint G[134]!=G[57]; 
constraint G[3]=G[41]; 
constraint G[66]!=G[8]; 
constraint G[40]!=G[149]; 
constraint G[33]!=G[148]; 
constraint G[105]!=G[84]; 
constraint G[17]=G[30]; 
constraint G[50]!=G[131]; 
constraint G[137]=G[120]; 
constraint G[43]!=G[122]; 
constraint G[8]=G[32]; 
constraint G[66]!=G[18]; 
constraint G[125]!=G[43]; 
constraint G[41]!=G[108]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) minimize Obj; 
output ["G = \(G)\nObj=\(Obj)"]; 
