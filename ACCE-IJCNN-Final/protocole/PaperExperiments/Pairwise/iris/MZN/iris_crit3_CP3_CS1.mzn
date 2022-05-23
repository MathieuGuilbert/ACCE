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

constraint G[150]=G[134]; 
constraint G[147]!=G[65]; 
constraint G[57]=G[81]; 
constraint G[98]=G[99]; 
constraint G[113]!=G[38]; 
constraint G[16]!=G[80]; 
constraint G[68]!=G[147]; 
constraint G[135]!=G[48]; 
constraint G[129]!=G[55]; 
constraint G[12]!=G[64]; 
constraint G[140]!=G[7]; 
constraint G[122]!=G[15]; 
constraint G[111]!=G[36]; 
constraint G[143]=G[133]; 
constraint G[79]!=G[113]; 
constraint G[41]=G[50]; 
constraint G[8]=G[24]; 
constraint G[114]!=G[95]; 
constraint G[48]!=G[94]; 
constraint G[26]!=G[107]; 
constraint G[134]!=G[67]; 
constraint G[32]!=G[100]; 
constraint G[96]!=G[39]; 
constraint G[5]=G[26]; 
constraint G[131]!=G[85]; 
constraint G[43]!=G[94]; 
constraint G[115]!=G[75]; 
constraint G[94]!=G[129]; 
constraint G[113]!=G[81]; 
constraint G[127]!=G[17]; 
constraint G[60]=G[79]; 
constraint G[119]!=G[62]; 
constraint G[1]!=G[128]; 
constraint G[145]=G[124]; 
constraint G[70]!=G[3]; 
constraint G[74]=G[55]; 
constraint G[145]!=G[39]; 
constraint G[81]!=G[106]; 
constraint G[79]!=G[43]; 
constraint G[146]!=G[23]; 
constraint G[22]=G[34]; 
constraint G[103]!=G[89]; 
constraint G[62]=G[94]; 
constraint G[147]=G[110]; 
constraint G[63]!=G[35]; 
constraint G[80]!=G[1]; 
constraint G[16]!=G[101]; 
constraint G[9]!=G[105]; 
constraint G[43]!=G[123]; 
constraint G[69]=G[55]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) maximize Obj; 
output ["G = \(G)\nObj=\(Obj)"]; 
