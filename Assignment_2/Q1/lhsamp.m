function S = lhsamp(n, m, lb, ub)
%
% Latin Hypercube Sampling
%
% [S] = lhsamp(n, m, lb, ub)
%
% INPUTS: 
%
% n : number of variables
% m : number of sample points to generate
% lb: lower bounds on variables
% ub: lower bounds on variables
%
% OUTPUT:
%
% S : the generated n dimensional m sample points chosen from
%     uniform distributions on m subdivions of the interval (lb, ub)
%     matrix of size n x m
%
% hbn@imm.dtu.dk  
% Last update April 12, 2002
%
% Updated August 2003 by PBN
% to incorporate upper and lower bounds on variables
%

   S = zeros(m,n);
   for i = 1 : n
       S(:, i) = (rand(1, m) + (randperm(m) - 1))' / m;
   end

   for i = 1 : n
       S(:,i) = lb(i) + S(:,i)*(ub(i) - lb(i));
   end;

   S = S';
