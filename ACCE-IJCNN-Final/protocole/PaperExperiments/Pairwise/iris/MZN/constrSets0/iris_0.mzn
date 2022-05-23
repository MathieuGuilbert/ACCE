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

constraint G[62]!=G[140]; 
constraint G[40]!=G[133]; 
constraint G[70]!=G[108]; 
constraint G[82]!=G[116]; 
constraint G[81]=G[62]; 
constraint G[1]!=G[92]; 
constraint G[62]!=G[35]; 
constraint G[120]!=G[5]; 
constraint G[100]=G[69]; 
constraint G[125]!=G[58]; 
constraint G[5]!=G[96]; 
constraint G[102]=G[121]; 
constraint G[130]!=G[54]; 
constraint G[19]!=G[150]; 
constraint G[44]!=G[72]; 
constraint G[4]!=G[80]; 
constraint G[83]=G[87]; 
constraint G[38]!=G[95]; 
constraint G[117]!=G[37]; 
constraint G[100]!=G[116]; 
constraint G[147]=G[126]; 
constraint G[117]!=G[85]; 
constraint G[117]!=G[17]; 
constraint G[147]=G[130]; 
constraint G[51]=G[84]; 
constraint G[15]!=G[53]; 
constraint G[112]=G[109]; 
constraint G[13]!=G[99]; 
constraint G[73]=G[83]; 
constraint G[17]!=G[64]; 
constraint G[82]!=G[138]; 
constraint G[112]!=G[17]; 
constraint G[39]=G[21]; 
constraint G[104]=G[127]; 
constraint G[63]!=G[125]; 
constraint G[41]=G[29]; 
constraint G[146]=G[113]; 
constraint G[37]!=G[79]; 
constraint G[84]!=G[22]; 
constraint G[111]!=G[83]; 
constraint G[79]!=G[111]; 
constraint G[138]=G[126]; 
constraint G[98]!=G[19]; 
constraint G[107]=G[138]; 
constraint G[42]=G[13]; 
constraint G[133]!=G[72]; 
constraint G[148]=G[105]; 
constraint G[93]!=G[30]; 
constraint G[80]!=G[112]; 
constraint G[54]=G[88]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) maximize S; 
output ["G = \(G)\nObj=\(S)"]; 
