include "globals.mzn"; 

int: n; % number of points 
int: k_min; 
int: k_max; 

array[1..n, 1..k_max] of float: M; 
array[1..n] of int: P; 
array[1..n] of var 1..k_max: G; 

var min(i in 1..n, k in 1..k_max)(M[i,k]) .. max(i in 1..n, k in 1..k_max)(M[i,k]) : Obj; 

constraint forall (i in 1..n, k in 1..k_max where P[i]=k) ( M[i,k] > Obj -> P[i]=G[i]); 
constraint G[19]!=G[105]; 
constraint G[186]!=G[76]; 
constraint G[85]!=G[204]; 
constraint G[19]!=G[142]; 
constraint G[35]!=G[182]; 
constraint G[168]!=G[118]; 
constraint G[173]!=G[137]; 
constraint G[151]!=G[74]; 
constraint G[132]=G[121]; 
constraint G[138]!=G[67]; 
constraint G[180]!=G[111]; 
constraint G[13]!=G[132]; 
constraint G[118]!=G[211]; 
constraint G[19]!=G[212]; 
constraint G[171]!=G[10]; 
constraint G[47]!=G[187]; 
constraint G[150]!=G[179]; 
constraint G[182]!=G[166]; 
constraint G[175]!=G[27]; 
constraint G[79]=G[89]; 
constraint G[51]!=G[109]; 
constraint G[133]!=G[29]; 
constraint G[112]=G[143]; 
constraint G[170]!=G[205]; 
constraint G[27]=G[41]; 
constraint G[23]=G[59]; 
constraint G[31]=G[44]; 
constraint G[28]!=G[186]; 
constraint G[29]=G[64]; 
constraint G[119]=G[84]; 
constraint G[178]!=G[27]; 
constraint G[33]!=G[144]; 
constraint G[100]=G[116]; 
constraint G[21]!=G[177]; 
constraint G[110]!=G[69]; 
constraint G[30]!=G[186]; 
constraint G[98]!=G[202]; 
constraint G[96]!=G[33]; 
constraint G[57]!=G[105]; 
constraint G[32]!=G[179]; 
constraint G[73]!=G[159]; 
constraint G[131]!=G[53]; 
constraint G[89]=G[133]; 
constraint G[91]!=G[70]; 
constraint G[72]=G[97]; 
constraint G[71]=G[125]; 
constraint G[189]!=G[33]; 
constraint G[160]=G[163]; 
constraint G[79]=G[112]; 
constraint G[15]!=G[197]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) minimize Obj; 
output ["G = \(G)\nObj=\(Obj)"]; 
