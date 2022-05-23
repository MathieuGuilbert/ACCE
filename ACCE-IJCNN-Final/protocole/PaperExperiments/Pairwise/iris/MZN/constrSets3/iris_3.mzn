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

constraint G[95]!=G[121]; 
constraint G[115]!=G[80]; 
constraint G[3]=G[30]; 
constraint G[83]!=G[138]; 
constraint G[121]!=G[58]; 
constraint G[73]=G[92]; 
constraint G[97]=G[85]; 
constraint G[68]!=G[40]; 
constraint G[95]!=G[101]; 
constraint G[2]=G[16]; 
constraint G[93]=G[63]; 
constraint G[128]!=G[32]; 
constraint G[88]!=G[25]; 
constraint G[17]!=G[93]; 
constraint G[46]!=G[56]; 
constraint G[112]!=G[12]; 
constraint G[16]!=G[59]; 
constraint G[29]!=G[63]; 
constraint G[17]!=G[94]; 
constraint G[130]=G[121]; 
constraint G[117]=G[146]; 
constraint G[46]!=G[100]; 
constraint G[56]=G[97]; 
constraint G[137]=G[130]; 
constraint G[96]!=G[27]; 
constraint G[32]!=G[72]; 
constraint G[38]=G[50]; 
constraint G[43]=G[19]; 
constraint G[20]!=G[97]; 
constraint G[22]!=G[125]; 
constraint G[12]!=G[141]; 
constraint G[132]!=G[5]; 
constraint G[8]!=G[138]; 
constraint G[123]!=G[47]; 
constraint G[48]=G[39]; 
constraint G[128]=G[137]; 
constraint G[24]=G[44]; 
constraint G[76]=G[62]; 
constraint G[19]!=G[101]; 
constraint G[92]!=G[105]; 
constraint G[132]!=G[28]; 
constraint G[114]!=G[3]; 
constraint G[28]!=G[136]; 
constraint G[92]!=G[137]; 
constraint G[7]!=G[99]; 
constraint G[113]!=G[96]; 
constraint G[109]!=G[72]; 
constraint G[17]=G[40]; 
constraint G[3]!=G[122]; 
constraint G[11]!=G[79]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) maximize S; 
output ["G = \(G)\nObj=\(S)"]; 
