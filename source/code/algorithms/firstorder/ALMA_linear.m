function [classifier, error_count, run_time, mistakes] = PA_linear(Y, X, options, id_list)
%--------------------------------------------------------------------------
t_tick = options.t_tick;

C = sqrt(2);
A = 0.9;
B = 1 / A;
p = 2;
k = 1;

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

	if(hat_y_t ~= y_t)
		error_count = error_count + 1;
	end

  gamma_k = B*sqrt(p-1) / sqrt(k);
	l_t = (1-A)*gamma_k - y_t*f_t;

	if(l_t > 0)
		eta_k = C / (sqrt(p-1)*sqrt(k));
		w = w + eta_k*y_t*x_t;
		norm_w = norm(w);
		w = w / (max(1,norm_w));
		k = k + 1;
	end

  if(mod(t, t_tick) == 0)
		mistakes = [mistakes error_count/t];
	end

end

classifier.w = w;

run_time = toc;
