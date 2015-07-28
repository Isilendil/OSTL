function [classifier, error_count, run_time, mistakes] = PA_I_OGD_linear(Y, X, Sim, options, id_list)
%--------------------------------------------------------------------------
C = options.C;
K = options.K;
m = options.m;
eta = options.eta;
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

  sim_vec = Sim(id,:);
	[sim, index] = sort(sim_vec, 'descend');
	index = index + m;
	for i = 1 : K
	  x_i = X(index(i),:);
		y_i = y_t;
		f_i = w * x_i';
		if(f_i >= 0)
			hat_y_i = 1;
		else
			hat_y_i = -1;
		end
		l_i = max(0, 1-y_i*f_i);
		if(l_i > 0)
			%eta_i = sim(i);
			%eta_i = min(0.001, sim(i));
			eta_i = eta;
			w = w + eta_i*y_i*x_i;
	  end
	end

  if(mod(t, t_tick) == 0)
		mistakes = [mistakes error_count/t];
	end

end

classifier.w = w;

run_time = toc;
