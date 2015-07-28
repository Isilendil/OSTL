
m = 1000;

ID = zeros(100, m);

for i = 1 : 100
	ID(i,:) = randperm(m);
end

save('../ID', 'ID');
