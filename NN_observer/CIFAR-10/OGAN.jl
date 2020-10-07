using Images, MLDatasets, ImageCore
using Flux, Statistics, CUDA
using Base.Iterators: partition
using Random
using Flux.Losses: logitbinarycrossentropy
using Flux.Optimise
using ProgressBars
using ImageIO
using ImageMagick
using LinearAlgebra
import Distributions: Uniform
include("hparams.jl")
include("networks.jl")
#using CUDAnative, CUDA, CUDAapi

#if has_cuda()
#    @info "CUDA is on"
    #CUDA.allowscalar(false)
#end

hparams = HPARAMS()

# load the training dataset
train_x, _ = MLDatasets.CIFAR10.traindata(Float32);
# load the test dataset
tester, _ = MLDatasets.CIFAR10.testdata(Float32);

data_x = [train_x[:,:,:,r] for r in partition(1:40000, hparams.batch_size)];
test_x = [tester[:,:,:,r] for r in partition(1:hparams.len_test, hparams.batch_size)];
#data_y = [train_y[r] for r in partition(1:40000,64)];

#=
#
#  Loss functions
#
=#

function d_loss(real_output, fake_output)
	real_loss = log(mean(logitbinarycrossentropy.(
					real_output,
					cu(ones(size(real_output))) |> gpu)))
	fake_loss = log(mean(logitbinarycrossentropy.(
					fake_output,
					cu(zeros(size(fake_output))) |> gpu)))

	return real_loss + fake_loss
end

function g_loss(real_output, fake_output)
	real_loss = log(mean(logitbinarycrossentropy.(
					real_output,
					cu(zeros(size(real_output))) |> gpu)))
	fake_loss = log(mean(logitbinarycrossentropy.(
					fake_output,
					cu(ones(size(fake_output))) |> gpu)))

	return real_loss + fake_loss
end

function o_loss(obs_real, obs_fake, control_factor)
	loss = 0.0

	real_loss = log(mean(logitbinarycrossentropy.(
					obs_real, zeros(size(obs_real)) |> gpu)))
	fake_loss = log(mean(logitbinarycrossentropy.(
					obs_fake, ones(size(obs_fake)) |> gpu)))
	loss = control_factor * abs(real_loss + fake_loss)
	return loss 
end
#=
#
# Lipschitz stuff
#
=# 

# K-Lipschitz condition
function lipschitz_bound(control_factor,observer, real_preimage, fake_preimage)
	diff_1 = mean(logitbinarycrossentropy.(real_preimage,
										   ones(size(real_preimage)) |> gpu))
	diff_2 = mean(logitbinarycrossentropy.(fake_preimage,
										   zeros(size(fake_preimage)) |> gpu))
	discriminator_distance = abs(diff_1^2 + diff_2^2)

	# assert homeomorphic properties on the observer
	# i.e. cts, 1 -> 1, 0 -> 0
	real_observation =  observer(real_preimage)
	false_observation =  observer(fake_preimage)
	obs_ones = ones(size(false_observation)) |> gpu
	obs_zeros = zeros(size(real_observation)) |> gpu

	diff_1 = log1p(mean(logitbinarycrossentropy.(real_observation, obs_zeros)))
	diff_2 = log1p(mean(logitbinarycrossentropy.(false_observation, obs_ones)))
	observer_distance = (diff_1 + diff_2)

	return discriminator_distance < abs(control_factor * observer_distance)
end

function lipschitz_switch(gen,dscr,obs,control_factor, x, hparams::HPARAMS)
	#noise = cu(randn!(
	#		similar(x, (hparams.latent_dim, hparams.batch_size)))) |> gpu
	noise = cu(randn(size(x))) |> gpu 
	real_preimage = dscr(x) |> gpu
	fake_preimage = dscr(gen(noise)) |> gpu

	switch = lipschitz_bound(control_factor,obs, real_preimage, fake_preimage)
	return switch
end
#=
#
#	Separate training modules
#
=#

function make_observations!(gen,dscr,obs, x, opt_dscr,opt_gen,opt_obs,
                             hparams::HPARAMS,control_factor,
                             rng::MersenneTwister, controller::Bool=true)
     counter_in_loop = 0
     loss_obs = 0.0

     rr = hparams.reduction_rate
     lc = hparams.loop_condition
     while controller
         # Now we will train the observer
         #noise = cu(randn!(
		 #similar(x, (hparams.latent_dim, hparams.batch_size)))) |> gpu
		 noise = cu(randn(size(x))) |> gpu
		 fakers = gen(noise)

         x = x |> gpu
         # data from the discriminator
         real_preimage = dscr(x)
         fake_preimage = dscr(fakers)
         # stuff from the observer
         real_obs = obs(real_preimage)
         fake_obs = obs(fake_preimage)

         # now take the parameters of the observer
         ps_obs = Flux.params(obs)
         loss_obs, back_obs = Flux.pullback(ps_obs) do
             o_loss(real_obs, fake_obs, control_factor)
         end
		 grad_obs = back_obs(1f0)
		 update!(opt_obs, ps_obs, grad_obs)
		 #noise = cu(randn!(
				#similar(x,(hparams.latent_dim, hparams.batch_size)))) |> gpu
		 noise = cu(randn(size(x)))|>gpu 
		 fake_input = gen(noise)
		 ps_dscr = Flux.params(dscr)

		 #Taking gradient
		 loss_dscr, back_dscr = Flux.pullback(ps_dscr) do
			 d_loss(dscr(fake_input), dscr(fake_input))
		 end
		 grad_dscr = back_dscr(1f0)
		 update!(opt_dscr, ps_dscr, grad_dscr)
		 
		 # apply the Lipschitz condition
		 real_preimage =  dscr(x) |> gpu
		 fake_preimage =  dscr(gen(noise)) |> gpu
		 controller = !lipschitz_bound(control_factor, obs,
								real_preimage,fake_preimage)

		 counter_in_loop += 1

		 if counter_in_loop > hparams.loop_condition * hparams.multiplier_lc
			 controller = false
			 if hparams.loop_condition > hparams.reduction_limit
				 hparams.reduction_rate = rr*(1.0 + randn(1)[1])
			 end
		 elseif counter_in_loop > hparams.loop_condition
			 # train the generator when the loop_condition is violated
			 loss_gen = train_generator!(gen,dscr,x,opt_gen, hparams)
			 end
		 end
return loss_obs, control_factor
end

# Train discriminator
function train_discriminator!(gen, dscr, x, opt_dscr, hparams::HPARAMS)
	# initialize
	loss_obs = 0.0
	noise = cu(randn(size(x))) |> gpu #randn!(similar(x, (hparams.latent_dim, hparams.batch_size))) |>gpu
	fake_input = gen(noise) |> gpu
	ps = Flux.params(dscr)
	#Taking gradient
	loss, back = Flux.pullback(ps) do
		d_loss(dscr(x)|>gpu, dscr(fake_input)|>gpu)
	end
	grad = back(1f0)
	update!(opt_dscr, ps, grad)

	return loss
end


# Train Generator (and the observer...)
function train_generator!(gen, dscr, x, opt_gen, hparams::HPARAMS)
	noise = cu(randn(size(x))) |> gpu #randn!(
			#similar(x, (hparams.latent_dim, hparams.batch_size))) |> gpu

	# Taking gradient
		control_factor = randn(1)[1] |>gpu
		loss = 0.0
		noise = cu(randn(size(x))) |>gpu #randn!(
				#similar(x, (hparams.latent_dim, hparams.batch_size))) |> gpu

		ps = Flux.params(gen) |> gpu
		dscr_x = dscr(x)
		loss, back = Flux.pullback(ps) do
			g_loss(dscr(gen(noise)), dscr_x)
		end
		grad = back(1f0)
		update!(opt_gen, ps, grad)

		return loss
end

#=
#
#	Main Train Module
#
=#

function train(; kws...)
	hparams = HPARAMS(;kws...)
	if ispath("./logs")
		rm("./logs", recursive=true)
		mkpath("./logs")
	else
		mkpath("./logs")
	end

	if ispath("./output")
		rm("./output", recursive=true)
		mkpath("./output")
	else
		mkpath("./output")
	end

	# Path to the logger
	a = "logs/logs.csv"
	first_line = "Step, Discriminator loss, Generator loss, Observer loss, Control factor\n"
	io = open(a,"a")
	write(io, first_line)
	# load the training dataset
	train_x, _ = MLDatasets.CIFAR10.traindata(Float32);
	# load the test dataset
	test_x, _ = MLDatasets.CIFAR10.testdata(Float32);
	data = [train_x[:,:,:,r] for r in partition(1:40000, 64)];
	#data_y = [train_y[r] for r in partition(1:40000,64)

	fixed_noise = [randn(hparams.latent_dim, 1) |> gpu
					for _=1:hparams.output_x*hparams.output_y]

	# Discriminator
	dscr = Discriminator()
	dscr = dscr |> gpu
	# Generator
	gen =  Generator(hparams)
	gen = gen |> gpu
	# Observer
	obs = Observer()
	obs = obs |> gpu

	# Optimizers
	opt_dscr = Optimiser(ExpDecay(),ADAM(hparams.lr_dscr))
	opt_gen = Optimiser(ExpDecay(),ADAM(hparams.lr_gen))
	opt_obs = Optimiser(ExpDecay(),ADAM(hparams.lr_obs))
	# Training
	train_steps = 0
	switch::Bool = false
	loss_obs = 0.0
	rng = MersenneTwister(1234)
	control_factor = rand(Uniform(0,hparams.max_lipschitz_bound),1)[1]
	for ep in 1:hparams.epochs
		@info "Epoch $ep"
		for y in ProgressBar(data)
			x = cu(y)
			# add noise on x every hparams.noise_addition steps
			if train_steps % hparams.noise_addition == 0
				x = x .* cu(randn(size(x)))
			elseif train_steps % hparams.noise_addition == 2
				x = x .+ cu(randn(size(x)))
			end

			# train the discriminator / observer first
			loss_dscr = train_discriminator!(gen, dscr, x, opt_dscr, hparams)
			# then train the generator
			loss_gen = train_generator!(gen, dscr, x, opt_gen, hparams)

			# this will determine whether we'd need to make observations or not
			if train_steps > hparams.start_observations
				#switch = lipschitz_switch(gen,dscr,obs,control_factor, x, hparams)
				switch = !switch
			end

			# now make the observations if needed
			if switch
				loss_obs, control_factor = make_observations!(
												gen,dscr,obs, x,
												opt_dscr,opt_gen,opt_obs,
												hparams,control_factor,
												rng, switch)
				loss_gen = train_generator!(gen,dscr,x,opt_gen, hparams)
			end

			# print the losses and the control_factor to a csv file
			if mod(train_steps, 0:10) == 0

				printer = "$(train_steps) , $(loss_dscr) , $(loss_gen) , $(loss_obs) , $(control_factor) \n"
				write(io, printer)
			end

			# output the sample images
			if train_steps % hparams.verbose_freq == 0
				@info("Train step $(train_steps), Discriminator loss = $(loss_dscr), Generator loss = $(loss_gen), Observer loss = $(loss_obs)")
				#@info("Test the discriminator")
				#test_val = 0.0
				#for a in ProgressBar(test_x)
				#	println(size(a))
				#	exit()
				#	noise = cu(randn(size(a))) |> gpu
				#	fake_input = gen(noise)
				#	test_d_loss = d_loss(dscr(a),dscr(fake_input))
				#	test_val += test_d_loss
				#end
				#a = test_x[:,:,:,mod(train_steps, 9999)]
				#noise = cu(randn(size(a))) |> gpu
				#fake = gen(noise)
				#test_val = d_loss(dscr(a),dscr(fake_input))
				#println("The test result: ",test_val)
				# Save generated fake image
				#output_image = create_output_image(gen, fixed_noise, hparams)
				#output_filename = "output/step_$(train_steps).png"
				#println(output_filename)
				#save(output_filename, output_image)
			end
			train_steps += 1
		end
	end
	close(io) # closing the log file
	#output_image = create_output_image(gen, fixed_noise, hparams)
	#save("output/last_step_$(train_steps).png", output_image)
end

#
#	Run the train function
#
cd(@__DIR__)
train()

