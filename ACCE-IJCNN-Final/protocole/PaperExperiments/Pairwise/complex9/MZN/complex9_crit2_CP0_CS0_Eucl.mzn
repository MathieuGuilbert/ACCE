include "globals.mzn"; 

int: n; % number of points 
int: k_min; 
int: k_max; 

array[1..n, 1..k_max] of int: M; 
array[1..n] of int: P; 
array[1..n] of var 1..k_max: G; 

var min(M) .. max(M) : Obj; 

constraint forall (i in 1..n, k in 1..k_max where P[i]=k) ( M[i,k] > Obj -> P[i]=G[i]); 


constraint G[2621]=G[2512]; 
constraint G[1739]!=G[107]; 
constraint G[1357]!=G[2210]; 
constraint G[2110]!=G[2962]; 
constraint G[2725]!=G[1244]; 
constraint G[2317]!=G[2028]; 
constraint G[274]!=G[2740]; 
constraint G[693]!=G[270]; 
constraint G[2420]!=G[1939]; 
constraint G[2893]!=G[387]; 
constraint G[66]!=G[2144]; 
constraint G[885]!=G[1711]; 
constraint G[645]=G[668]; 
constraint G[452]!=G[1652]; 
constraint G[2842]!=G[2129]; 
constraint G[1235]!=G[2792]; 
constraint G[264]!=G[763]; 
constraint G[921]!=G[1322]; 
constraint G[1688]!=G[2258]; 
constraint G[173]!=G[2622]; 
constraint G[2521]!=G[220]; 
constraint G[2857]!=G[1096]; 
constraint G[1585]!=G[217]; 
constraint G[2040]=G[1999]; 
constraint G[227]!=G[2946]; 
constraint G[2100]!=G[281]; 
constraint G[1354]!=G[389]; 
constraint G[1512]!=G[629]; 
constraint G[2269]!=G[1002]; 
constraint G[1868]!=G[2553]; 
constraint G[1096]!=G[630]; 
constraint G[1814]!=G[2344]; 
constraint G[1695]!=G[2928]; 
constraint G[2659]!=G[214]; 
constraint G[1321]!=G[2255]; 
constraint G[2496]!=G[1878]; 
constraint G[2306]!=G[987]; 
constraint G[1229]=G[1223]; 
constraint G[349]!=G[996]; 
constraint G[259]!=G[1207]; 
constraint G[1459]=G[1370]; 
constraint G[1582]=G[1709]; 
constraint G[2935]!=G[939]; 
constraint G[927]!=G[2027]; 
constraint G[2753]=G[2976]; 
constraint G[1372]!=G[2384]; 
constraint G[308]!=G[1332]; 
constraint G[2304]!=G[1891]; 
constraint G[706]!=G[986]; 
constraint G[358]!=G[1449]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) minimize Obj; 
output ["G = \(G)\nObj=\(Obj)"]; 
