include "globals.mzn"; 

int: n; % number of points 
int: k_min; 
int: k_max; 
int: a=1; 

array[1..n, 1..k_max] of float: M; 
array[1..n] of int: P; 
array[1..n] of var 1..k_max: G; 

var min(i in 1..n, k in 1..k_max)(M[i,k]) .. max(i in 1..n, k in 1..k_max)(M[i,k]) : Obj; 

constraint forall (i in 1..n, k in 1..k_max where G[i]=k) ( M[i,k] < Obj -> P[i]=G[i]); array[1..n] of var -1..max(i in 1..n, k in 1..k_max)(M[i,k]): is_moved; 
constraint forall (i in 1..n)(is_moved[i]=-1 -> P[i]=G[i]); 
constraint forall (i in 1..n, k in 1..k_max where P[i]=k)(P[i]!=G[i] -> is_moved[i]=M[i,k]); 
predicate atleast(int: a, array[int] of var int: is_moved, int: Obj) ; 

constraint G[89]!=G[126]; 
constraint G[105]!=G[80]; 
constraint G[119]!=G[61]; 
constraint G[53]=G[57]; 
constraint G[93]=G[55]; 
constraint G[22]=G[32]; 
constraint G[146]=G[107]; 
constraint G[2]!=G[78]; 
constraint G[117]!=G[58]; 
constraint G[135]!=G[4]; 
constraint G[150]!=G[99]; 
constraint G[23]=G[5]; 
constraint G[23]!=G[147]; 
constraint G[122]!=G[29]; 
constraint G[60]=G[81]; 
constraint G[116]!=G[96]; 
constraint G[70]!=G[30]; 
constraint G[14]!=G[127]; 
constraint G[46]!=G[74]; 
constraint G[23]!=G[73]; 
constraint G[127]=G[135]; 
constraint G[9]!=G[116]; 
constraint G[21]!=G[57]; 
constraint G[15]!=G[137]; 
constraint G[96]!=G[120]; 
constraint G[97]=G[80]; 
constraint G[12]!=G[131]; 
constraint G[96]=G[59]; 
constraint G[105]=G[101]; 
constraint G[138]!=G[56]; 
constraint G[115]!=G[57]; 
constraint G[56]=G[89]; 
constraint G[119]=G[124]; 
constraint G[58]!=G[29]; 
constraint G[103]!=G[86]; 
constraint G[132]!=G[71]; 
constraint G[11]=G[13]; 
constraint G[83]!=G[4]; 
constraint G[18]!=G[145]; 
constraint G[126]!=G[47]; 
constraint G[73]!=G[40]; 
constraint G[86]=G[59]; 
constraint G[122]!=G[48]; 
constraint G[99]=G[79]; 
constraint G[5]!=G[82]; 
constraint G[21]=G[20]; 
constraint G[25]!=G[111]; 
constraint G[30]=G[12]; 
constraint G[116]!=G[50]; 
constraint G[146]!=G[49]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) maximize Obj; 
output ["G = \(G)\nObj=\(Obj)"]; 
