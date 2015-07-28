function [classifier, error_count, run_time, mistakes] = OGD_linear(Y, X, options, id_list)
%--------------------------------------------------------------------------
t_tick = options.t_tick;

ID = id_list;
error_count = 0;
mistakes = [];

eta = 1;
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

	if(hat_y_t ~= y_t)
		error_count = error_count + 1;
	end

	l_t = max(0, 1-y_t*f_t);
	if(l_t > 0)
		eta_t = eta / sqrt(t);
	  w = w + eta_t*y_t*x_t;
	end


  if(mod(t, t_tick) == 0)
		mistakes = [mistakes error_count/t];
	end

end

classifier.w = w;

run_time = toc;
