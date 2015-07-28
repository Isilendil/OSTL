function [classifier, error_count, run_time, mistakes] = OGD_OGD(Y, Kernel, Sim, options, id_list)
%--------------------------------------------------------------------------
C = options.C;
K = options.K;
m = options.m;
eta = options.eta;
t_tick = options.t_tick;

alpha = [];
SV = [];
ID = id_list;
error_count = 0;
mistakes = [];

% loop
tic

for t = 1 : length(ID)
	id = ID(t);
  y_t = Y(id);

	if(isempty(alpha))
		f_t = 0;
	else
		k_t = Kernel(id, SV(:))';
		f_t = alpha * k_t;
	end

	l_t = max(0, 1-y_t*f_t);
	hat_y_t = sign(f_t);
	if(hat_y_t == 0)
		hat_y_t = 1;
	end

	if(hat_y_t ~= y_t)
		error_count = error_count + 1;
	end

	if(l_t > 0)
		eta_t = eta / sqrt(t);
		alpha = [alpha y_t*eta_t;];
		SV = [SV id];
	end

  sim_vec = Sim(id,:);
	[sim, index] = sort(sim_vec, 'descend');
	index = index + m;
	for i = 1 : K
		y_i = y_t;
		if(isempty(alpha))
			f_i = 0;
		else
			k_i = Kernel(index(i), SV(:))';
			f_i = alpha * k_i;
		end
		l_i = max(0, 1-y_i*f_i);
		hat_y_i = sign(f_i);
		if(hat_y_i == 0)
			hat_y_i = 1;
		end
	  if(l_i > 0)
			%eta_i = sim(i);
			%eta_i = min(0.001, sim(i));
			eta_i = eta;
			alpha = [alpha y_i*eta_i;];
			SV = [SV index(i)];
		end
	end


  if(mod(t, t_tick) == 0)
		mistakes = [mistakes error_count/t];
	end

end

classifier.SV = SV;
classifier.alpha = alpha;

run_time = toc;

