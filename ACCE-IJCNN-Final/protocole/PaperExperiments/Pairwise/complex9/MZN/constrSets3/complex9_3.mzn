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

constraint G[932]!=G[1266]; 
constraint G[1643]!=G[2265]; 
constraint G[328]!=G[2360]; 
constraint G[104]!=G[564]; 
constraint G[1684]!=G[138]; 
constraint G[1469]!=G[2039]; 
constraint G[1560]!=G[258]; 
constraint G[800]!=G[1993]; 
constraint G[2890]=G[2768]; 
constraint G[983]=G[1001]; 
constraint G[624]!=G[2140]; 
constraint G[1141]!=G[555]; 
constraint G[2733]!=G[1476]; 
constraint G[2985]!=G[1637]; 
constraint G[2266]!=G[122]; 
constraint G[1024]!=G[1399]; 
constraint G[2387]!=G[392]; 
constraint G[981]!=G[2448]; 
constraint G[2525]!=G[1872]; 
constraint G[1201]=G[1339]; 
constraint G[1811]!=G[2938]; 
constraint G[2766]!=G[2236]; 
constraint G[1895]=G[1663]; 
constraint G[808]!=G[2925]; 
constraint G[804]!=G[596]; 
constraint G[2929]=G[3031]; 
constraint G[785]!=G[1840]; 
constraint G[342]!=G[916]; 
constraint G[489]!=G[1254]; 
constraint G[1011]!=G[520]; 
constraint G[1192]!=G[2722]; 
constraint G[1072]=G[1288]; 
constraint G[407]!=G[2926]; 
constraint G[1126]!=G[1977]; 
constraint G[2409]!=G[1739]; 
constraint G[1427]!=G[2925]; 
constraint G[2862]!=G[608]; 
constraint G[1728]!=G[383]; 
constraint G[2993]!=G[1233]; 
constraint G[2995]!=G[340]; 
constraint G[60]=G[41]; 
constraint G[776]!=G[2662]; 
constraint G[446]!=G[1778]; 
constraint G[571]!=G[233]; 
constraint G[2773]!=G[436]; 
constraint G[2864]!=G[351]; 
constraint G[220]!=G[1875]; 
constraint G[1505]!=G[426]; 
constraint G[2048]=G[2091]; 
constraint G[2405]!=G[1776]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) maximize S; 
output ["G = \(G)\nObj=\(S)"]; 
