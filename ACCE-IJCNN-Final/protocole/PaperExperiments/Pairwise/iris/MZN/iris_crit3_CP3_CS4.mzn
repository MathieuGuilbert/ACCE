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

constraint G[85]!=G[8]; 
constraint G[48]!=G[104]; 
constraint G[58]!=G[144]; 
constraint G[18]!=G[108]; 
constraint G[8]=G[19]; 
constraint G[44]!=G[71]; 
constraint G[33]!=G[95]; 
constraint G[53]=G[62]; 
constraint G[109]!=G[50]; 
constraint G[24]!=G[100]; 
constraint G[126]!=G[96]; 
constraint G[30]!=G[51]; 
constraint G[101]!=G[65]; 
constraint G[36]!=G[73]; 
constraint G[33]!=G[147]; 
constraint G[42]!=G[66]; 
constraint G[55]!=G[44]; 
constraint G[132]!=G[92]; 
constraint G[58]!=G[36]; 
constraint G[65]!=G[117]; 
constraint G[124]!=G[22]; 
constraint G[22]!=G[146]; 
constraint G[67]!=G[138]; 
constraint G[91]!=G[10]; 
constraint G[8]!=G[133]; 
constraint G[118]!=G[71]; 
constraint G[68]=G[84]; 
constraint G[74]=G[66]; 
constraint G[118]=G[102]; 
constraint G[17]!=G[69]; 
constraint G[61]!=G[30]; 
constraint G[124]!=G[59]; 
constraint G[120]!=G[1]; 
constraint G[75]!=G[142]; 
constraint G[115]=G[129]; 
constraint G[79]!=G[42]; 
constraint G[6]=G[11]; 
constraint G[125]!=G[36]; 
constraint G[85]=G[51]; 
constraint G[37]=G[9]; 
constraint G[93]!=G[11]; 
constraint G[60]!=G[133]; 
constraint G[133]!=G[85]; 
constraint G[87]=G[76]; 
constraint G[87]!=G[101]; 
constraint G[107]!=G[14]; 
constraint G[64]!=G[121]; 
constraint G[143]!=G[16]; 
constraint G[55]!=G[21]; 
constraint G[72]=G[86]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) maximize Obj; 
output ["G = \(G)\nObj=\(Obj)"]; 
