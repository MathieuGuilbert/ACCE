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

constraint G[585]!=G[1662]; 
constraint G[2859]!=G[575]; 
constraint G[1477]=G[1921]; 
constraint G[2053]!=G[2531]; 
constraint G[2889]!=G[600]; 
constraint G[18]!=G[1188]; 
constraint G[1233]!=G[379]; 
constraint G[675]!=G[2439]; 
constraint G[1483]!=G[1040]; 
constraint G[1951]!=G[2955]; 
constraint G[1142]=G[1549]; 
constraint G[161]!=G[1974]; 
constraint G[2980]!=G[1931]; 
constraint G[2262]!=G[1213]; 
constraint G[2561]=G[2811]; 
constraint G[371]!=G[257]; 
constraint G[2374]!=G[1282]; 
constraint G[2037]!=G[1051]; 
constraint G[2109]!=G[643]; 
constraint G[2828]=G[2923]; 
constraint G[2220]!=G[1879]; 
constraint G[2926]!=G[910]; 
constraint G[2099]!=G[2745]; 
constraint G[2372]=G[2948]; 
constraint G[2408]!=G[590]; 
constraint G[1106]!=G[836]; 
constraint G[1826]=G[1517]; 
constraint G[2890]!=G[2066]; 
constraint G[215]!=G[2155]; 
constraint G[3023]!=G[868]; 
constraint G[2284]!=G[933]; 
constraint G[1645]!=G[340]; 
constraint G[2946]!=G[694]; 
constraint G[61]!=G[1207]; 
constraint G[339]!=G[776]; 
constraint G[1272]!=G[2118]; 
constraint G[2630]!=G[1599]; 
constraint G[1577]!=G[1963]; 
constraint G[1533]!=G[2330]; 
constraint G[2169]!=G[1665]; 
constraint G[2096]!=G[924]; 
constraint G[956]=G[871]; 
constraint G[1814]=G[1070]; 
constraint G[971]!=G[477]; 
constraint G[1300]!=G[396]; 
constraint G[1533]!=G[722]; 
constraint G[240]!=G[146]; 
constraint G[59]!=G[2070]; 
constraint G[2499]=G[2738]; 
constraint G[97]!=G[431]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) maximize S; 
output ["G = \(G)\nObj=\(S)"]; 
