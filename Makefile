all:
	@make -C integrand_gamma_exp
	@make -C integrand_rq

clean:
	@make -C integrand_gamma_exp clean
	@make -C integrand_rq clean
