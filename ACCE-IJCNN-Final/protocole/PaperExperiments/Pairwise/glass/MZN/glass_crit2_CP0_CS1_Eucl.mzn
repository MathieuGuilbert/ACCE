include "globals.mzn"; 

int: n; % number of points 
int: k_min; 
int: k_max; 

array[1..n, 1..k_max] of int: M; 
array[1..n] of int: P; 
array[1..n] of var 1..k_max: G; 

var min(M) .. max(M) : Obj; 

constraint forall (i in 1..n, k in 1..k_max where P[i]=k) ( M[i,k] > Obj -> P[i]=G[i]); 


constraint G[67]!=G[192]; 
constraint G[93]!=G[30]; 
constraint G[141]!=G[40]; 
constraint G[118]=G[110]; 
constraint G[95]!=G[16]; 
constraint G[64]!=G[87]; 
constraint G[141]!=G[5]; 
constraint G[187]!=G[77]; 
constraint G[80]=G[94]; 
constraint G[65]=G[31]; 
constraint G[74]!=G[168]; 
constraint G[151]!=G[197]; 
constraint G[26]=G[36]; 
constraint G[5]=G[4]; 
constraint G[129]!=G[16]; 
constraint G[6]!=G[89]; 
constraint G[165]!=G[184]; 
constraint G[146]!=G[160]; 
constraint G[161]!=G[50]; 
constraint G[8]!=G[145]; 
constraint G[35]!=G[188]; 
constraint G[11]!=G[93]; 
constraint G[188]!=G[163]; 
constraint G[34]!=G[171]; 
constraint G[132]=G[79]; 
constraint G[181]!=G[6]; 
constraint G[195]=G[206]; 
constraint G[177]!=G[81]; 
constraint G[116]!=G[191]; 
constraint G[78]!=G[35]; 
constraint G[134]=G[72]; 
constraint G[14]!=G[74]; 
constraint G[81]!=G[153]; 
constraint G[92]!=G[176]; 
constraint G[181]!=G[147]; 
constraint G[92]=G[120]; 
constraint G[179]!=G[188]; 
constraint G[24]!=G[164]; 
constraint G[188]!=G[130]; 
constraint G[25]!=G[79]; 
constraint G[47]!=G[73]; 
constraint G[199]!=G[111]; 
constraint G[95]!=G[31]; 
constraint G[52]!=G[200]; 
constraint G[17]!=G[194]; 
constraint G[73]=G[134]; 
constraint G[170]!=G[26]; 
constraint G[135]!=G[190]; 
constraint G[189]=G[200]; 
constraint G[132]!=G[183]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) minimize Obj; 
output ["G = \(G)\nObj=\(Obj)"]; 
