include "globals.mzn"; 

int: n; % number of points 
int: k_min; 
int: k_max; 

array[1..n, 1..k_max] of int: M; 
array[1..n] of int: P; 
array[1..n] of var 1..k_max: G; 

var min(M) .. max(M) : Obj; 

constraint forall (i in 1..n, k in 1..k_max where P[i]=k) ( M[i,k] > Obj -> P[i]=G[i]); 


constraint G[1479]=G[1138]; 
constraint G[1185]=G[1824]; 
constraint G[1813]!=G[2471]; 
constraint G[395]=G[494]; 
constraint G[614]=G[646]; 
constraint G[1372]!=G[897]; 
constraint G[715]!=G[2239]; 
constraint G[1131]!=G[767]; 
constraint G[1871]!=G[223]; 
constraint G[96]!=G[2373]; 
constraint G[1282]!=G[2789]; 
constraint G[24]!=G[291]; 
constraint G[651]!=G[180]; 
constraint G[807]!=G[19]; 
constraint G[1251]!=G[2426]; 
constraint G[654]!=G[2112]; 
constraint G[1567]!=G[2355]; 
constraint G[499]!=G[2803]; 
constraint G[121]!=G[2089]; 
constraint G[1812]!=G[3001]; 
constraint G[498]!=G[1140]; 
constraint G[2696]!=G[1333]; 
constraint G[2215]!=G[2691]; 
constraint G[764]!=G[2199]; 
constraint G[1429]!=G[2288]; 
constraint G[197]!=G[954]; 
constraint G[1253]=G[1688]; 
constraint G[602]=G[682]; 
constraint G[1357]!=G[2617]; 
constraint G[2258]!=G[3013]; 
constraint G[485]!=G[694]; 
constraint G[653]!=G[2927]; 
constraint G[431]!=G[239]; 
constraint G[2195]!=G[1821]; 
constraint G[2384]!=G[2157]; 
constraint G[2311]=G[2642]; 
constraint G[304]!=G[29]; 
constraint G[1551]!=G[731]; 
constraint G[1121]!=G[2132]; 
constraint G[745]!=G[700]; 
constraint G[831]!=G[1446]; 
constraint G[1039]!=G[2273]; 
constraint G[2633]!=G[1916]; 
constraint G[1250]!=G[197]; 
constraint G[2677]!=G[258]; 
constraint G[2671]!=G[1346]; 
constraint G[2317]!=G[2235]; 
constraint G[127]!=G[2422]; 
constraint G[2905]!=G[130]; 
constraint G[691]=G[589]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) minimize Obj; 
output ["G = \(G)\nObj=\(Obj)"]; 
