include "globals.mzn"; 

int: n; % number of points 
int: k_min; 
int: k_max; 

array[1..n, 1..k_max] of float: M; 
array[1..n] of int: P; 
array[1..n] of var 1..k_max: G; 

var 0..n: S; 

constraint S=sum(i in 1..n, k in 1..k_max where G[i]=k)(M[i,k]); 
constraint G[144]=G[126]; 
constraint G[144]!=G[70]; 
constraint G[42]!=G[149]; 
constraint G[27]!=G[110]; 
constraint G[64]!=G[149]; 
constraint G[37]!=G[87]; 
constraint G[13]!=G[59]; 
constraint G[92]!=G[136]; 
constraint G[89]=G[98]; 
constraint G[41]=G[23]; 
constraint G[28]!=G[147]; 
constraint G[136]=G[121]; 
constraint G[17]!=G[133]; 
constraint G[95]!=G[24]; 
constraint G[145]!=G[23]; 
constraint G[9]!=G[70]; 
constraint G[50]!=G[90]; 
constraint G[80]=G[71]; 
constraint G[36]=G[26]; 
constraint G[12]=G[17]; 
constraint G[74]!=G[28]; 
constraint G[6]!=G[82]; 
constraint G[76]!=G[128]; 
constraint G[136]=G[147]; 
constraint G[86]=G[84]; 
constraint G[2]!=G[136]; 
constraint G[79]=G[75]; 
constraint G[2]!=G[75]; 
constraint G[90]!=G[112]; 
constraint G[22]!=G[93]; 
constraint G[14]!=G[113]; 
constraint G[104]!=G[33]; 
constraint G[30]!=G[81]; 
constraint G[51]!=G[36]; 
constraint G[41]!=G[106]; 
constraint G[125]=G[142]; 
constraint G[52]!=G[135]; 
constraint G[14]!=G[126]; 
constraint G[19]!=G[128]; 
constraint G[94]!=G[145]; 
constraint G[91]=G[64]; 
constraint G[36]!=G[90]; 
constraint G[95]!=G[107]; 
constraint G[129]=G[110]; 
constraint G[132]!=G[78]; 
constraint G[40]=G[39]; 
constraint G[96]!=G[27]; 
constraint G[87]!=G[150]; 
constraint G[77]!=G[139]; 
constraint G[6]!=G[120]; 
%%%%%%%%%%%%% 

solve ::int_search(G, first_fail, indomain_min, complete) maximize S; 
output ["G = \(G)\nObj=\(S)"]; 
