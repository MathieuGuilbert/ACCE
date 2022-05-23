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

constraint G[112]!=G[73]; 
constraint G[82]=G[59]; 
constraint G[8]!=G[125]; 
constraint G[113]=G[138]; 
constraint G[137]!=G[60]; 
constraint G[145]!=G[4]; 
constraint G[115]!=G[60]; 
constraint G[71]=G[65]; 
constraint G[40]=G[11]; 
constraint G[123]!=G[31]; 
constraint G[62]=G[96]; 
constraint G[82]=G[79]; 
constraint G[30]=G[21]; 
constraint G[100]!=G[147]; 
constraint G[105]!=G[6]; 
constraint G[74]=G[90]; 
constraint G[33]=G[24]; 
constraint G[110]!=G[8]; 
constraint G[79]!=G[114]; 
constraint G[51]!=G[14]; 
constraint G[124]=G[131]; 
constraint G[119]=G[139]; 
constraint G[63]!=G[3]; 
constraint G[31]!=G[83]; 
constraint G[81]=G[67]; 
constraint G[84]=G[72]; 
constraint G[49]=G[12]; 
constraint G[61]!=G[101]; 
constraint G[15]=G[16]; 
constraint G[37]=G[38]; 
constraint G[150]!=G[57]; 
constraint G[61]!=G[8]; 
constraint G[106]!=G[83]; 
constraint G[6]=G[29]; 
constraint G[50]!=G[149]; 
constraint G[2]!=G[103]; 
constraint G[118]!=G[1]; 
constraint G[52]=G[78]; 
constraint G[111]!=G[95]; 
constraint G[6]=G[18]; 
constraint G[38]!=G[78]; 
constraint G[112]=G[146]; 
constraint G[73]=G[96]; 
constraint G[67]=G[90]; 
constraint G[144]!=G[32]; 
constraint G[84]!=G[144]; 
constraint G[70]!=G[13]; 
constraint G[64]!=G[133]; 
constraint G[120]!=G[95]; 
constraint G[56]=G[88]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) maximize Obj; 
output ["G = \(G)\nObj=\(Obj)"]; 
