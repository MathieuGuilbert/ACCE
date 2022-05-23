include "globals.mzn"; 

int: n; % number of points 
int: k_min; 
int: k_max; 

array[1..n, 1..n] of float: dist; 
array[1..n] of var 1..k_max: G; 

var min(i,j in 1..n where i<j)(dist[i,j]) .. max(i,j in 1..n where i<j)(dist[i,j]) : S; 

constraint forall (i,j in 1..n where i<j) ( dist[i,j] < S -> G[i] == G[j] ); 
constraint G[1] = 1; 
constraint value_precede_chain([i | i in 1..k_max], G); 
constraint max(G) >= k_min; 

constraint G[287]!=G[2227]; 
constraint G[1740]!=G[939]; 
constraint G[929]!=G[2518]; 
constraint G[249]!=G[2985]; 
constraint G[111]!=G[2372]; 
constraint G[240]!=G[1031]; 
constraint G[1454]=G[1133]; 
constraint G[2323]!=G[824]; 
constraint G[2856]!=G[963]; 
constraint G[1695]=G[1611]; 
constraint G[1924]=G[1315]; 
constraint G[1301]!=G[494]; 
constraint G[1526]!=G[95]; 
constraint G[2683]!=G[267]; 
constraint G[474]!=G[1934]; 
constraint G[2624]=G[2313]; 
constraint G[1233]!=G[2698]; 
constraint G[2809]!=G[992]; 
constraint G[780]=G[1031]; 
constraint G[2459]!=G[468]; 
constraint G[2172]!=G[471]; 
constraint G[1614]=G[1389]; 
constraint G[2633]!=G[1960]; 
constraint G[2184]!=G[1448]; 
constraint G[1372]!=G[90]; 
constraint G[2839]!=G[488]; 
constraint G[577]!=G[1833]; 
constraint G[975]!=G[2699]; 
constraint G[831]!=G[1324]; 
constraint G[1102]=G[1127]; 
constraint G[42]!=G[1604]; 
constraint G[154]!=G[726]; 
constraint G[308]!=G[2822]; 
constraint G[266]!=G[1365]; 
constraint G[2071]!=G[1375]; 
constraint G[309]!=G[2992]; 
constraint G[1269]!=G[2256]; 
constraint G[1748]=G[1168]; 
constraint G[572]!=G[2029]; 
constraint G[1236]!=G[680]; 
constraint G[500]!=G[2093]; 
constraint G[2509]!=G[1181]; 
constraint G[355]!=G[2197]; 
constraint G[533]!=G[1780]; 
constraint G[297]!=G[411]; 
constraint G[187]!=G[392]; 
constraint G[715]!=G[1548]; 
constraint G[530]!=G[2497]; 
constraint G[1033]!=G[1677]; 
constraint G[1389]=G[1061]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) maximize S; 
output ["G = \(G)\nObj=\(S)"]; 
