function [R] = top_right(N,n)
ss = size(N);
ii_row = 1:n;
ii_col = ss(2)-n+1 : ss(2);
%R = N(1:n:,end-n:end);
R = N(ii_row,ii_col);
end