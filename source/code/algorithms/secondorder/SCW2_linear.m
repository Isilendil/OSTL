function [classifier, error_count, run_time, mistakes] = SCW2_linear(Y, X, options, id_list)
%--------------------------------------------------------------------------
t_tick = options.t_tick;

ID = id_list;
error_count = 0;
mistakes = [];

C = 1;
A = 1;
Sigma = A * eye(size(X,2));
seed = 0.9;
phi = norminv(seed, 0, 1);
psi = 1 + (phi^2)/2;
xi = 1 + phi^2;
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

  v_t = x_t*Sigma*x_t';   % confidence
  m_t = y_t*f_t;          % margin
  n_t = v_t + 1/(2*C);
  l_t = phi*sqrt(v_t) - m_t; % loss
  if(l_t > 0)
    alpha_t = max(0,(-(2*m_t*n_t+phi^2*m_t*v_t) + sqrt(phi^4*m_t^2*v_t*2+4*n_t*v_t*phi^2*(n_t+v_t*phi*2)))/(2*(n_t^2+n_t*v_t*phi^2)));
    u_t     = 0.25*(-alpha_t*v_t*phi+sqrt(alpha_t^2*v_t^2*phi^2+4*v_t))^2;
    beta_t  = alpha_t*phi/(sqrt(u_t)+alpha_t*phi*v_t);
    S_x_t   = x_t*Sigma';
    w       = w + alpha_t*y_t*S_x_t;
    Sigma   = Sigma - beta_t*S_x_t'*S_x_t;
  end

  if(mod(t, t_tick) == 0)
		mistakes = [mistakes error_count/t];
	end

end

classifier.w = w;
classifier.Sigma = Sigma;

run_time = toc;
