include "globals.mzn"; 

int: n; % number of points 
int: k_min; 
int: k_max; 

array[1..n, 1..k_max] of int: M; 
array[1..n] of int: P; 
array[1..n] of var 1..k_max: G; 

var min(M) .. max(M) : Obj; 

constraint forall (i in 1..n, k in 1..k_max where P[i]=k) ( M[i,k] > Obj -> P[i]=G[i]); 


constraint G[218]!=G[65]; 
constraint G[238]!=G[588]; 
constraint G[39]=G[80]; 
constraint G[517]!=G[561]; 
constraint G[373]!=G[475]; 
constraint G[315]!=G[153]; 
constraint G[414]!=G[167]; 
constraint G[567]!=G[397]; 
constraint G[451]!=G[409]; 
constraint G[218]!=G[9]; 
constraint G[369]!=G[265]; 
constraint G[367]!=G[218]; 
constraint G[401]!=G[108]; 
constraint G[522]=G[530]; 
constraint G[395]!=G[256]; 
constraint G[418]!=G[101]; 
constraint G[400]!=G[101]; 
constraint G[502]!=G[347]; 
constraint G[252]!=G[426]; 
constraint G[210]!=G[158]; 
constraint G[516]=G[505]; 
constraint G[233]!=G[573]; 
constraint G[568]!=G[96]; 
constraint G[361]!=G[114]; 
constraint G[180]!=G[414]; 
constraint G[69]!=G[250]; 
constraint G[283]!=G[177]; 
constraint G[274]!=G[378]; 
constraint G[376]!=G[465]; 
constraint G[402]!=G[88]; 
constraint G[435]!=G[534]; 
constraint G[294]!=G[547]; 
constraint G[3]!=G[573]; 
constraint G[210]=G[199]; 
constraint G[5]!=G[193]; 
constraint G[92]!=G[238]; 
constraint G[400]!=G[372]; 
constraint G[118]!=G[180]; 
constraint G[456]!=G[539]; 
constraint G[463]=G[443]; 
constraint G[264]!=G[488]; 
constraint G[417]!=G[466]; 
constraint G[249]=G[235]; 
constraint G[573]!=G[131]; 
constraint G[125]!=G[328]; 
constraint G[169]!=G[271]; 
constraint G[85]!=G[198]; 
constraint G[88]!=G[349]; 
constraint G[432]!=G[105]; 
constraint G[573]!=G[27]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) minimize Obj; 
output ["G = \(G)\nObj=\(Obj)"]; 
