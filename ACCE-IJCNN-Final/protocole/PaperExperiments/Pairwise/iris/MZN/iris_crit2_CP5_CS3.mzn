include "globals.mzn"; 

int: n; % number of points 
int: k_min; 
int: k_max; 

array[1..n, 1..k_max] of int: M; 
array[1..n] of int: P; 
array[1..n] of var 1..k_max: G; 

var min(M) .. max(M) : Obj; 

constraint forall (i in 1..n, k in 1..k_max where P[i]=k) ( M[i,k] > Obj -> P[i]=G[i]); 


constraint G[9]!=G[98]; 
constraint G[90]=G[61]; 
constraint G[136]=G[124]; 
constraint G[22]!=G[136]; 
constraint G[118]!=G[29]; 
constraint G[74]=G[54]; 
constraint G[129]=G[110]; 
constraint G[106]!=G[20]; 
constraint G[42]!=G[79]; 
constraint G[83]=G[73]; 
constraint G[67]!=G[38]; 
constraint G[28]!=G[69]; 
constraint G[49]=G[6]; 
constraint G[19]=G[3]; 
constraint G[90]!=G[30]; 
constraint G[96]=G[65]; 
constraint G[21]!=G[121]; 
constraint G[75]=G[62]; 
constraint G[141]!=G[42]; 
constraint G[98]!=G[6]; 
constraint G[39]=G[46]; 
constraint G[48]=G[10]; 
constraint G[142]!=G[80]; 
constraint G[4]!=G[91]; 
constraint G[22]=G[32]; 
constraint G[113]=G[140]; 
constraint G[31]!=G[85]; 
constraint G[126]=G[122]; 
constraint G[104]=G[133]; 
constraint G[115]!=G[27]; 
constraint G[132]!=G[51]; 
constraint G[70]!=G[121]; 
constraint G[50]=G[13]; 
constraint G[131]=G[123]; 
constraint G[137]=G[119]; 
constraint G[43]!=G[86]; 
constraint G[149]!=G[49]; 
constraint G[55]!=G[125]; 
constraint G[62]!=G[115]; 
constraint G[20]=G[50]; 
constraint G[81]!=G[122]; 
constraint G[33]!=G[139]; 
constraint G[99]!=G[140]; 
constraint G[138]!=G[6]; 
constraint G[115]=G[111]; 
constraint G[75]!=G[146]; 
constraint G[86]!=G[40]; 
constraint G[112]!=G[8]; 
constraint G[122]!=G[19]; 
constraint G[137]!=G[82]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) minimize Obj; 
output ["G = \(G)\nObj=\(Obj)"]; 
