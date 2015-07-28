function [classifier, error_count, run_time, mistakes] = PA_I_linear(Y, X, options, id_list)
%--------------------------------------------------------------------------
C = options.C;
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

	l_t = max(0, 1-y_t*f_t);
	if(hat_y_t ~= y_t)
		error_count = error_count + 1;
	end

	if(l_t > 0)
		s_t = norm(x_t) ^ 2;
		tau_t = min(C, l_t/s_t);
	  w = w + tau_t*y_t*x_t;
	end


  if(mod(t, t_tick) == 0)
		mistakes = [mistakes error_count/t];
	end

end

classifier.w = w;

run_time = toc;
