include "globals.mzn"; 

int: n; % number of points 
int: k_min; 
int: k_max; 

array[1..n, 1..k_max] of int: M; 
array[1..n] of int: P; 
array[1..n] of var 1..k_max: G; 

var min(M) .. max(M) : Obj; 

constraint forall (i in 1..n, k in 1..k_max where P[i]=k) ( M[i,k] > Obj -> P[i]=G[i]); 


constraint G[485]!=G[96]; 
constraint G[483]!=G[449]; 
constraint G[368]!=G[282]; 
constraint G[280]!=G[383]; 
constraint G[7]!=G[525]; 
constraint G[45]!=G[166]; 
constraint G[71]!=G[285]; 
constraint G[91]!=G[247]; 
constraint G[558]!=G[290]; 
constraint G[302]!=G[320]; 
constraint G[500]!=G[266]; 
constraint G[336]!=G[503]; 
constraint G[305]!=G[375]; 
constraint G[544]!=G[238]; 
constraint G[200]!=G[280]; 
constraint G[36]!=G[315]; 
constraint G[166]!=G[570]; 
constraint G[52]!=G[569]; 
constraint G[420]!=G[187]; 
constraint G[469]!=G[186]; 
constraint G[549]!=G[322]; 
constraint G[116]!=G[36]; 
constraint G[538]!=G[394]; 
constraint G[138]!=G[1]; 
constraint G[101]!=G[159]; 
constraint G[137]!=G[588]; 
constraint G[202]!=G[412]; 
constraint G[72]!=G[250]; 
constraint G[341]!=G[311]; 
constraint G[142]!=G[368]; 
constraint G[75]!=G[561]; 
constraint G[362]=G[384]; 
constraint G[498]!=G[60]; 
constraint G[553]!=G[311]; 
constraint G[99]!=G[473]; 
constraint G[163]!=G[262]; 
constraint G[345]!=G[96]; 
constraint G[250]!=G[149]; 
constraint G[420]!=G[554]; 
constraint G[339]!=G[207]; 
constraint G[112]!=G[459]; 
constraint G[171]!=G[581]; 
constraint G[89]=G[101]; 
constraint G[430]!=G[577]; 
constraint G[436]!=G[232]; 
constraint G[106]!=G[296]; 
constraint G[509]!=G[218]; 
constraint G[297]!=G[358]; 
constraint G[574]!=G[284]; 
constraint G[111]!=G[575]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) minimize Obj; 
output ["G = \(G)\nObj=\(Obj)"]; 
