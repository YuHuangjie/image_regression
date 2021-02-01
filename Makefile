all:
	@make -C integrand_gamma_exp
	@make -C integrand_rq
	@make -C integrand_poly

clean:
	@make -C integrand_gamma_exp clean
	@make -C integrand_rq clean
	@make -C integrand_poly clean
