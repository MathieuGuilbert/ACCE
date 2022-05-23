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

constraint G[1347]!=G[332]; 
constraint G[23]!=G[2013]; 
constraint G[1637]!=G[2211]; 
constraint G[2177]!=G[1958]; 
constraint G[1078]!=G[2039]; 
constraint G[2938]!=G[2218]; 
constraint G[1346]=G[1633]; 
constraint G[1903]!=G[75]; 
constraint G[2067]!=G[2599]; 
constraint G[2521]!=G[2180]; 
constraint G[2554]!=G[1109]; 
constraint G[2804]!=G[801]; 
constraint G[750]!=G[1562]; 
constraint G[251]!=G[969]; 
constraint G[2823]=G[2798]; 
constraint G[167]!=G[265]; 
constraint G[128]!=G[1793]; 
constraint G[2285]!=G[1775]; 
constraint G[2802]!=G[476]; 
constraint G[1612]!=G[2061]; 
constraint G[2149]!=G[1573]; 
constraint G[2113]=G[2217]; 
constraint G[912]=G[1041]; 
constraint G[2536]=G[2529]; 
constraint G[584]!=G[2120]; 
constraint G[724]=G[899]; 
constraint G[2134]!=G[1671]; 
constraint G[545]!=G[2067]; 
constraint G[1444]!=G[53]; 
constraint G[1177]=G[1659]; 
constraint G[100]!=G[647]; 
constraint G[775]=G[872]; 
constraint G[1344]!=G[141]; 
constraint G[2344]!=G[678]; 
constraint G[1504]=G[1291]; 
constraint G[1565]=G[1355]; 
constraint G[2955]!=G[1827]; 
constraint G[1679]!=G[2913]; 
constraint G[2316]!=G[1598]; 
constraint G[151]!=G[2373]; 
constraint G[2579]!=G[518]; 
constraint G[2867]!=G[1472]; 
constraint G[82]!=G[1340]; 
constraint G[2654]!=G[1163]; 
constraint G[659]!=G[1868]; 
constraint G[2563]!=G[1793]; 
constraint G[2819]!=G[1255]; 
constraint G[307]!=G[639]; 
constraint G[2824]!=G[675]; 
constraint G[1799]!=G[2301]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) maximize S; 
output ["G = \(G)\nObj=\(S)"]; 
