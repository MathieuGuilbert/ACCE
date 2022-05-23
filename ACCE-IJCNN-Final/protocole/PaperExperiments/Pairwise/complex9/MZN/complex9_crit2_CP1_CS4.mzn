include "globals.mzn"; 

int: n; % number of points 
int: k_min; 
int: k_max; 

array[1..n, 1..k_max] of int: M; 
array[1..n] of int: P; 
array[1..n] of var 1..k_max: G; 

var min(M) .. max(M) : Obj; 

constraint forall (i in 1..n, k in 1..k_max where P[i]=k) ( M[i,k] > Obj -> P[i]=G[i]); 


constraint G[180]!=G[2563]; 
constraint G[2750]!=G[1516]; 
constraint G[1644]=G[1309]; 
constraint G[966]=G[868]; 
constraint G[2971]=G[2540]; 
constraint G[1079]=G[1552]; 
constraint G[1730]!=G[2773]; 
constraint G[112]!=G[2696]; 
constraint G[2004]!=G[1917]; 
constraint G[1715]!=G[2823]; 
constraint G[1995]!=G[2372]; 
constraint G[520]!=G[490]; 
constraint G[859]!=G[1447]; 
constraint G[2302]!=G[428]; 
constraint G[1354]=G[1868]; 
constraint G[2669]=G[2952]; 
constraint G[3008]=G[2522]; 
constraint G[596]!=G[1551]; 
constraint G[152]!=G[669]; 
constraint G[1603]=G[1490]; 
constraint G[442]!=G[2732]; 
constraint G[2177]!=G[2915]; 
constraint G[2742]=G[2950]; 
constraint G[502]!=G[21]; 
constraint G[777]!=G[1703]; 
constraint G[427]!=G[1583]; 
constraint G[2540]!=G[2035]; 
constraint G[2603]!=G[767]; 
constraint G[1664]!=G[2974]; 
constraint G[1797]!=G[1979]; 
constraint G[821]!=G[2710]; 
constraint G[2517]!=G[1353]; 
constraint G[2290]!=G[695]; 
constraint G[1556]!=G[550]; 
constraint G[86]!=G[229]; 
constraint G[1525]=G[1818]; 
constraint G[2084]!=G[597]; 
constraint G[1773]=G[1138]; 
constraint G[1345]!=G[64]; 
constraint G[1396]!=G[911]; 
constraint G[504]!=G[1914]; 
constraint G[2315]!=G[1684]; 
constraint G[2496]!=G[1877]; 
constraint G[2902]!=G[406]; 
constraint G[336]!=G[1450]; 
constraint G[1739]!=G[2879]; 
constraint G[2172]!=G[1422]; 
constraint G[2788]!=G[1981]; 
constraint G[57]!=G[2109]; 
constraint G[2512]!=G[795]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) minimize Obj; 
output ["G = \(G)\nObj=\(Obj)"]; 
