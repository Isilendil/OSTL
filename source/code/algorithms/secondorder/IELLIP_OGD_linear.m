function [classifier, error_count, run_time, mistakes] = IELLIP_linear(Y, X, Sim, options, id_list)
%--------------------------------------------------------------------------
t_tick = options.t_tick;

ID = id_list;
error_count = 0;
mistakes = [];

eta = options.eta;
m = options.m;
K = options.K;

A = 1;
Sigma = A * eye(size(X,2));
b = 0.3;
c_t = 0.1;
w = zeros(1, size(X,2));
% loop
tic

for t = 1 : length(ID)
	id = ID(t);
  x_t = X(id,:);
	y_t = Y(id);

  f_t = w*x_t';
  if (f_t>=0)
    hat_y_t = 1;
  else
    hat_y_t = -1;
  end

	if(hat_y_t ~= y_t)
		error_count = error_count + 1;
	end

  l_t = (hat_y_t ~= y_t); % 0 - correct prediction, 1 - incorrect
  v_t = x_t*Sigma*x_t';   % confidence
  m_t = y_t*f_t;          % margin
  if (l_t > 0),
    if v_t ~= 0,
        alpha_t = (1-m_t)/sqrt(v_t);
        g_t     = y_t*x_t/sqrt(v_t);
        S_x_t   = g_t*Sigma';
        w       = w + alpha_t*S_x_t;
        Sigma   = (Sigma - c_t*S_x_t'*S_x_t)/(1-c_t);
    end
  end

  c_t = c_t * b;

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
classifier.Sigma = Sigma;

run_time = toc;
