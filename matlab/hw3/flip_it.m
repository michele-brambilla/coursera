function [w] = flip_it(v)

m = length(v);
w = zeros(size(v));
for index = 1:m
    w(m - index + 1) = v(index);
end

end