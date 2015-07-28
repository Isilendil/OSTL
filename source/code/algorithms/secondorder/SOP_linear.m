function [classifier, error_count, run_time, mistakes] = SOP_linear(Y, X, options, id_list)
%--------------------------------------------------------------------------
t_tick = options.t_tick;

ID = id_list;
error_count = 0;
mistakes = [];

A = 1;
Sigma = A * eye(size(X,2));

w = zeros(1, size(X,2));
% loop
tic

for t = 1 : length(ID)
	id = ID(t);
  x_t = X(id,:);
	y_t = Y(id);

  S_x_t   = x_t*Sigma';
  v_t     = x_t*S_x_t';
  beta_t  = 1/(v_t+1);
  Sigma_t = Sigma - beta_t*S_x_t'*S_x_t;
  f_t     = w*Sigma_t*x_t';

	if(f_t >= 0)
		hat_y_t = 1;
	else
		hat_y_t = -1;
	end

	if(hat_y_t ~= y_t)
		error_count = error_count + 1;
	end

  l_t = (hat_y_t ~= y_t); 
  if (l_t > 0),    
    w = w + y_t*x_t;    
  end
	Sigma = Sigma_t;

  if(mod(t, t_tick) == 0)
		mistakes = [mistakes error_count/t];
	end

end

classifier.w = w;
classifier.Sigma = Sigma;

run_time = toc;
