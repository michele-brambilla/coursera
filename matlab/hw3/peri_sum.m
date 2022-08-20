function [perimeter] = peri_sum(A)

    perimeter = sum(A(:,1)) + sum(A(:,end));
    perimeter = perimeter + sum(A(1,2:end-1)) + sum(A(end,2:end-1));

end