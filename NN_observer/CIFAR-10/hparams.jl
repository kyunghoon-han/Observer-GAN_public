using Parameters

@with_kw mutable struct HPARAMS
	batch_size::Int = 4
	latent_dim::Int = 10
	epochs::Int = 1000
	len_test = 10000
	# feedback frequency
	verbose_freq::Int = 1000
	# output size for image creation
	output_x::Int = 6
	output_y::Int = 6
	# learning rates
	lr_dscr::Float64 = 0.00005
	lr_gen::Float64 = 0.0005
	lr_obs::Float64 = 0.00002
	# max observation loop size = loop_condition * multiplier_lc
	loop_condition::Float64 = 5.0
	multiplier_lc::Int = 3
	# start observations from this step
	start_observations::Int = 400
	# adjust the maximum size of the range
	max_lipschitz_bound::Float64 = 5.0
	# multiply noise on the real input every this steps
	noise_addition::Int = 100
	# multiplier on max_lipschitz_bound
	# if the observation goes on until it hits the maximum allowed 
	reduction_rate::Float64 = 0.95
	# minimum allowed K value for K-Lipschitz condition
	reduction_limit::Float64 = 1.0
end
hparams = HPARAMS
