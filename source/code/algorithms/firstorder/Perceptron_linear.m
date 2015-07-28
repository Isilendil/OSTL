function [classifier, error_count, run_time, mistakes] = Perceptron_linear(Y, X, options, id_list)
%--------------------------------------------------------------------------
t_tick = options.t_tick;

ID = id_list;
error_count = 0;
mistakes = [];

w = zeros(1, size(X,2));
% loop
tic

for t = 1 : length(ID)
	id = ID(t);
  x_t = X(id,:);
	y_t = Y(id);

  f_t = w * x_t';
	if(f_t >= 0)
		hat_y_t = 1;
	else
		hat_y_t = -1;
	end

	l_t = (hat_y_t ~= y_t);
	if(l_t > 0)
		error_count = error_count + 1;
		w = w + y_t*x_t;
	end

  if(mod(t, t_tick) == 0)
		mistakes = [mistakes error_count/t];
	end

end

classifier.w = w;

run_time = toc;
