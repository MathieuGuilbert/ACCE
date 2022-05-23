include "globals.mzn"; 

int: n; % number of points 
int: k_min; 
int: k_max; 

array[1..n, 1..k_max] of int: M; 
array[1..n] of int: P; 
array[1..n] of var 1..k_max: G; 

var min(M) .. max(M) : Obj; 

constraint forall (i in 1..n, k in 1..k_max where P[i]=k) ( M[i,k] > Obj -> P[i]=G[i]); 


constraint G[2642]!=G[1785]; 
constraint G[98]!=G[2809]; 
constraint G[1856]!=G[2149]; 
constraint G[2623]!=G[1156]; 
constraint G[434]!=G[1188]; 
constraint G[113]!=G[1594]; 
constraint G[2995]!=G[1538]; 
constraint G[2436]!=G[1799]; 
constraint G[2561]!=G[684]; 
constraint G[2592]!=G[1643]; 
constraint G[2812]!=G[994]; 
constraint G[1797]=G[1957]; 
constraint G[715]!=G[1499]; 
constraint G[2187]=G[1994]; 
constraint G[1534]!=G[421]; 
constraint G[445]!=G[721]; 
constraint G[2963]!=G[877]; 
constraint G[2750]=G[2378]; 
constraint G[2575]!=G[205]; 
constraint G[1110]!=G[7]; 
constraint G[2984]=G[2505]; 
constraint G[2053]!=G[1167]; 
constraint G[1122]!=G[2090]; 
constraint G[743]!=G[169]; 
constraint G[1391]!=G[953]; 
constraint G[1692]!=G[27]; 
constraint G[1757]=G[1139]; 
constraint G[1380]=G[1598]; 
constraint G[1528]!=G[3026]; 
constraint G[2685]!=G[662]; 
constraint G[1230]!=G[2167]; 
constraint G[2467]!=G[619]; 
constraint G[1832]!=G[982]; 
constraint G[3010]!=G[1123]; 
constraint G[678]!=G[2062]; 
constraint G[103]!=G[2285]; 
constraint G[1105]!=G[2463]; 
constraint G[2236]!=G[1135]; 
constraint G[2076]!=G[2578]; 
constraint G[1390]=G[1560]; 
constraint G[1751]!=G[2791]; 
constraint G[1748]!=G[2982]; 
constraint G[1295]!=G[2027]; 
constraint G[20]!=G[1262]; 
constraint G[865]!=G[1543]; 
constraint G[1185]!=G[2363]; 
constraint G[2055]!=G[356]; 
constraint G[1577]=G[1612]; 
constraint G[1528]!=G[669]; 
constraint G[402]!=G[753]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) minimize Obj; 
output ["G = \(G)\nObj=\(Obj)"]; 
