function [classifier, error_count, run_time, mistakes] = NAROW_linear(Y, X, Sim, options, id_list)
%--------------------------------------------------------------------------
t_tick = options.t_tick;

ID = id_list;
error_count = 0;
mistakes = [];

eta = options.eta;
m = options.m;
K = options.K;

C = options.C;
A = 1;
b = C;
Sigma = A * eye(size(X,2));

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

  v_t = x_t*Sigma*x_t'; % confidence
  m_t = y_t*f_t;        % margin 
  l_t = 1 - m_t;        % hinge loss
  if l_t > 0,
    chi_t = x_t*Sigma*x_t'; % inv(A_{t-1}^{-1})?
    if chi_t > 1/b,
        r_t = chi_t/(b*chi_t-1);
    else
        r_t = inf;
    end
    beta_t  = 1/(v_t + r_t);
    alpha_t = max(0, 1-m_t)*beta_t;
    S_x_t   = x_t*Sigma';
    w       = w + alpha_t*y_t*S_x_t;
    Sigma   = Sigma - beta_t*S_x_t'*S_x_t;
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
