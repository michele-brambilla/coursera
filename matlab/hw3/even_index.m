function [N] = even_index(M)

p = size(M);
N = M(2:2:p(1),2:2:p(2));

end