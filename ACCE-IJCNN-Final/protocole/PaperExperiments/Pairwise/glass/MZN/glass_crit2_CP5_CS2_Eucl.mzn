include "globals.mzn"; 

int: n; % number of points 
int: k_min; 
int: k_max; 

array[1..n, 1..k_max] of int: M; 
array[1..n] of int: P; 
array[1..n] of var 1..k_max: G; 

var min(M) .. max(M) : Obj; 

constraint forall (i in 1..n, k in 1..k_max where P[i]=k) ( M[i,k] > Obj -> P[i]=G[i]); 


constraint G[108]=G[86]; 
constraint G[59]!=G[159]; 
constraint G[9]!=G[187]; 
constraint G[92]=G[90]; 
constraint G[105]!=G[154]; 
constraint G[175]=G[169]; 
constraint G[175]!=G[24]; 
constraint G[22]!=G[82]; 
constraint G[53]!=G[101]; 
constraint G[43]!=G[155]; 
constraint G[76]!=G[13]; 
constraint G[175]!=G[7]; 
constraint G[31]!=G[133]; 
constraint G[114]!=G[152]; 
constraint G[150]!=G[194]; 
constraint G[123]=G[84]; 
constraint G[116]!=G[12]; 
constraint G[176]!=G[157]; 
constraint G[28]!=G[184]; 
constraint G[17]!=G[102]; 
constraint G[1]!=G[132]; 
constraint G[174]=G[176]; 
constraint G[49]!=G[105]; 
constraint G[14]!=G[118]; 
constraint G[198]!=G[117]; 
constraint G[172]!=G[69]; 
constraint G[15]=G[31]; 
constraint G[165]!=G[59]; 
constraint G[138]=G[146]; 
constraint G[161]!=G[68]; 
constraint G[148]!=G[103]; 
constraint G[48]!=G[131]; 
constraint G[107]!=G[164]; 
constraint G[53]=G[15]; 
constraint G[204]!=G[51]; 
constraint G[150]!=G[21]; 
constraint G[64]!=G[91]; 
constraint G[150]!=G[48]; 
constraint G[193]=G[190]; 
constraint G[51]!=G[114]; 
constraint G[92]!=G[164]; 
constraint G[209]!=G[74]; 
constraint G[97]!=G[186]; 
constraint G[93]=G[92]; 
constraint G[130]!=G[200]; 
constraint G[183]!=G[124]; 
constraint G[30]=G[68]; 
constraint G[74]!=G[174]; 
constraint G[152]!=G[11]; 
constraint G[105]=G[145]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) minimize Obj; 
output ["G = \(G)\nObj=\(Obj)"]; 
