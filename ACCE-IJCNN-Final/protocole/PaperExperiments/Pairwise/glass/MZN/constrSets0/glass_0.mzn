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

constraint G[177]!=G[97]; 
constraint G[79]!=G[32]; 
constraint G[131]!=G[59]; 
constraint G[190]!=G[141]; 
constraint G[122]!=G[47]; 
constraint G[209]!=G[12]; 
constraint G[43]!=G[186]; 
constraint G[167]!=G[83]; 
constraint G[102]!=G[20]; 
constraint G[40]!=G[128]; 
constraint G[123]!=G[38]; 
constraint G[81]=G[121]; 
constraint G[93]!=G[207]; 
constraint G[72]!=G[161]; 
constraint G[146]!=G[2]; 
constraint G[32]!=G[202]; 
constraint G[198]=G[196]; 
constraint G[21]!=G[209]; 
constraint G[84]=G[95]; 
constraint G[191]!=G[43]; 
constraint G[143]=G[118]; 
constraint G[105]!=G[193]; 
constraint G[132]=G[140]; 
constraint G[105]!=G[54]; 
constraint G[181]=G[182]; 
constraint G[59]=G[8]; 
constraint G[72]=G[121]; 
constraint G[149]!=G[207]; 
constraint G[162]!=G[76]; 
constraint G[4]!=G[116]; 
constraint G[27]=G[61]; 
constraint G[148]=G[159]; 
constraint G[54]!=G[78]; 
constraint G[38]!=G[163]; 
constraint G[107]!=G[149]; 
constraint G[16]!=G[191]; 
constraint G[27]!=G[102]; 
constraint G[8]!=G[146]; 
constraint G[184]!=G[60]; 
constraint G[206]=G[210]; 
constraint G[94]!=G[177]; 
constraint G[52]!=G[150]; 
constraint G[198]!=G[63]; 
constraint G[126]!=G[7]; 
constraint G[66]!=G[154]; 
constraint G[119]!=G[170]; 
constraint G[212]!=G[178]; 
constraint G[154]!=G[34]; 
constraint G[35]!=G[174]; 
constraint G[104]=G[96]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) maximize S; 
output ["G = \(G)\nObj=\(S)"]; 
