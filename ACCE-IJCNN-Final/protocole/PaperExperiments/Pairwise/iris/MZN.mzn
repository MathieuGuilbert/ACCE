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

constraint G[39]=G[4]; 
constraint G[95]!=G[7]; 
constraint G[57]!=G[106]; 
constraint G[2]!=G[133]; 
constraint G[55]!=G[147]; 
constraint G[92]!=G[104]; 
constraint G[142]!=G[31]; 
constraint G[120]!=G[29]; 
constraint G[86]!=G[10]; 
constraint G[124]!=G[54]; 
constraint G[130]!=G[57]; 
constraint G[29]=G[44]; 
constraint G[58]=G[65]; 
constraint G[94]=G[74]; 
constraint G[96]=G[95]; 
constraint G[67]=G[78]; 
constraint G[143]!=G[89]; 
constraint G[32]=G[15]; 
constraint G[136]!=G[59]; 
constraint G[112]!=G[84]; 
constraint G[111]!=G[86]; 
constraint G[9]!=G[52]; 
constraint G[70]=G[53]; 
constraint G[128]!=G[54]; 
constraint G[98]!=G[12]; 
constraint G[142]!=G[91]; 
constraint G[101]!=G[32]; 
constraint G[113]!=G[54]; 
constraint G[58]!=G[132]; 
constraint G[25]=G[40]; 
constraint G[138]!=G[49]; 
constraint G[145]!=G[35]; 
constraint G[148]!=G[65]; 
constraint G[62]!=G[37]; 
constraint G[106]=G[103]; 
constraint G[99]=G[72]; 
constraint G[19]!=G[98]; 
constraint G[80]!=G[103]; 
constraint G[33]=G[44]; 
constraint G[106]!=G[52]; 
constraint G[110]=G[117]; 
constraint G[144]!=G[32]; 
constraint G[41]!=G[146]; 
constraint G[137]!=G[14]; 
constraint G[74]!=G[124]; 
constraint G[125]!=G[97]; 
constraint G[40]!=G[150]; 
constraint G[111]=G[117]; 
constraint G[96]!=G[112]; 
constraint G[9]!=G[129]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) maximize Obj; 
output ["G = \(G)\nObj=\(Obj)"]; 
