function [classifier, error_count, run_time, mistakes] = OGD(Y, Kernel, options, id_list)
%--------------------------------------------------------------------------
t_tick = options.t_tick;

alpha = [];
SV = [];
ID = id_list;
error_count = 0;
mistakes = [];

eta = 1;
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


  if(mod(t, t_tick) == 0)
		mistakes = [mistakes error_count/t];
	end

end

classifier.SV = SV;
classifier.alpha = alpha;

run_time = toc;

