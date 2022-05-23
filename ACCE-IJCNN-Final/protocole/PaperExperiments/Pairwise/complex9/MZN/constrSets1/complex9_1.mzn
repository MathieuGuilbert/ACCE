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

constraint G[2090]!=G[373]; 
constraint G[17]!=G[2121]; 
constraint G[2774]!=G[1498]; 
constraint G[317]!=G[280]; 
constraint G[2938]!=G[1857]; 
constraint G[2615]!=G[1082]; 
constraint G[574]!=G[294]; 
constraint G[731]!=G[2923]; 
constraint G[1511]!=G[2709]; 
constraint G[1164]=G[1158]; 
constraint G[1161]!=G[2890]; 
constraint G[2399]!=G[74]; 
constraint G[507]!=G[2978]; 
constraint G[1951]=G[1639]; 
constraint G[1528]!=G[851]; 
constraint G[1098]!=G[2579]; 
constraint G[1110]!=G[863]; 
constraint G[3005]!=G[1271]; 
constraint G[656]=G[705]; 
constraint G[927]!=G[1879]; 
constraint G[199]!=G[2230]; 
constraint G[833]!=G[1375]; 
constraint G[1844]=G[1073]; 
constraint G[2199]!=G[374]; 
constraint G[2740]!=G[1601]; 
constraint G[1159]=G[1066]; 
constraint G[1712]!=G[772]; 
constraint G[769]!=G[2186]; 
constraint G[2555]!=G[821]; 
constraint G[1422]=G[1350]; 
constraint G[1483]!=G[2571]; 
constraint G[2561]=G[2323]; 
constraint G[1939]!=G[2725]; 
constraint G[2084]!=G[779]; 
constraint G[1769]!=G[630]; 
constraint G[347]!=G[2969]; 
constraint G[796]!=G[373]; 
constraint G[2858]=G[2764]; 
constraint G[2221]=G[2064]; 
constraint G[2367]=G[2697]; 
constraint G[2860]=G[2815]; 
constraint G[2964]!=G[1324]; 
constraint G[2001]!=G[1863]; 
constraint G[1790]!=G[2434]; 
constraint G[1904]=G[1581]; 
constraint G[1826]=G[1062]; 
constraint G[1999]!=G[2473]; 
constraint G[434]!=G[2440]; 
constraint G[1067]!=G[2560]; 
constraint G[2126]!=G[2740]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) maximize S; 
output ["G = \(G)\nObj=\(S)"]; 
