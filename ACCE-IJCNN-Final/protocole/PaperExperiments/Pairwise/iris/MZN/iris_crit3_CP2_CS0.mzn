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

constraint G[102]!=G[12]; 
constraint G[62]!=G[117]; 
constraint G[45]!=G[126]; 
constraint G[139]!=G[70]; 
constraint G[150]=G[129]; 
constraint G[6]!=G[96]; 
constraint G[48]=G[18]; 
constraint G[30]=G[31]; 
constraint G[31]!=G[127]; 
constraint G[138]!=G[85]; 
constraint G[139]=G[123]; 
constraint G[78]!=G[130]; 
constraint G[13]=G[1]; 
constraint G[141]!=G[5]; 
constraint G[19]=G[21]; 
constraint G[18]!=G[78]; 
constraint G[75]!=G[149]; 
constraint G[9]!=G[99]; 
constraint G[136]!=G[51]; 
constraint G[49]!=G[77]; 
constraint G[134]!=G[12]; 
constraint G[147]!=G[33]; 
constraint G[44]=G[5]; 
constraint G[9]!=G[60]; 
constraint G[75]=G[57]; 
constraint G[95]=G[87]; 
constraint G[32]=G[22]; 
constraint G[113]=G[135]; 
constraint G[102]!=G[96]; 
constraint G[55]!=G[105]; 
constraint G[82]!=G[33]; 
constraint G[17]!=G[79]; 
constraint G[125]=G[106]; 
constraint G[95]=G[76]; 
constraint G[88]!=G[27]; 
constraint G[60]!=G[5]; 
constraint G[42]=G[33]; 
constraint G[16]!=G[68]; 
constraint G[66]!=G[22]; 
constraint G[124]!=G[42]; 
constraint G[29]!=G[97]; 
constraint G[31]!=G[92]; 
constraint G[85]!=G[131]; 
constraint G[119]!=G[39]; 
constraint G[57]=G[79]; 
constraint G[36]!=G[134]; 
constraint G[48]!=G[90]; 
constraint G[13]=G[41]; 
constraint G[131]!=G[28]; 
constraint G[77]!=G[36]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) maximize Obj; 
output ["G = \(G)\nObj=\(Obj)"]; 
