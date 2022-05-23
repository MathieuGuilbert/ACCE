include "globals.mzn"; 

int: n; % number of points 
int: k_min; 
int: k_max; 

array[1..n, 1..k_max] of int: M; 
array[1..n] of int: P; 
array[1..n] of var 1..k_max: G; 

var min(M) .. max(M) : Obj; 

constraint forall (i in 1..n, k in 1..k_max where P[i]=k) ( M[i,k] > Obj -> P[i]=G[i]); 


constraint G[51]!=G[137]; 
constraint G[133]!=G[32]; 
constraint G[62]!=G[146]; 
constraint G[102]!=G[39]; 
constraint G[99]!=G[45]; 
constraint G[44]!=G[114]; 
constraint G[120]!=G[18]; 
constraint G[130]=G[106]; 
constraint G[49]!=G[80]; 
constraint G[58]!=G[121]; 
constraint G[50]!=G[148]; 
constraint G[141]!=G[10]; 
constraint G[121]!=G[13]; 
constraint G[50]!=G[64]; 
constraint G[96]!=G[6]; 
constraint G[100]!=G[40]; 
constraint G[48]!=G[90]; 
constraint G[33]!=G[141]; 
constraint G[127]!=G[43]; 
constraint G[47]=G[49]; 
constraint G[60]!=G[135]; 
constraint G[41]!=G[76]; 
constraint G[132]!=G[35]; 
constraint G[19]!=G[125]; 
constraint G[26]!=G[68]; 
constraint G[61]!=G[133]; 
constraint G[125]!=G[30]; 
constraint G[18]=G[24]; 
constraint G[63]=G[64]; 
constraint G[142]!=G[5]; 
constraint G[139]=G[119]; 
constraint G[59]!=G[50]; 
constraint G[122]=G[112]; 
constraint G[145]!=G[41]; 
constraint G[2]=G[39]; 
constraint G[129]!=G[25]; 
constraint G[17]=G[21]; 
constraint G[54]=G[66]; 
constraint G[19]=G[2]; 
constraint G[122]!=G[46]; 
constraint G[57]!=G[106]; 
constraint G[126]=G[132]; 
constraint G[86]!=G[106]; 
constraint G[10]!=G[136]; 
constraint G[4]!=G[146]; 
constraint G[85]=G[76]; 
constraint G[18]=G[46]; 
constraint G[23]!=G[139]; 
constraint G[119]!=G[98]; 
constraint G[18]=G[9]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) minimize Obj; 
output ["G = \(G)\nObj=\(Obj)"]; 
