using Base.Iterators: partition
using CuArrays
using Flux
using Flux: logitbinarycrossentropy, mse
using Flux.Optimise
using Images
using MLDatasets
using Statistics
using Parameters
using Printf
using Random
using ProgressBars
using ImageIO
using ImageMagick
using LinearAlgebra
import Distributions: Uniform

@with_kw mutable struct HyperParams
	batch_size::Int = 128
	latent_dim::Int = 100
	epochs::Int = 1000
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

# Load MNIST dataset
images, _ = MLDatasets.MNIST.traindata(Float32)
# # Normalize the dataset to [-1,1]
image_tensor = reshape(@.(2f0 * images - 1f0),28,28,1,:);

function create_output_image(gen, fixed_noise, hparams)
	@eval Flux.istraining() = false
	fake_images = @. cpu(gen(fixed_noise))
	@eval Flux.istraining() = true
	a = reduce(vcat, reduce.(hcat, partition(fake_images, hparams.output_y)))
	a = dropdims(a ; dims = (3,4))
	image_array = permutedims(a, (2, 1))
	image_array = @. Gray(a + 1f0) / 2f0
	return image_array
end

function Discriminator()
    return Chain(
            Conv((4, 4), 1 => 64; stride = 2, pad = 1),
            x->leakyrelu.(x, 0.2f0),
            Dropout(0.25),
			Conv((4, 4), 64 => 128; stride = 2, pad = 1),
            x->leakyrelu.(x, 0.2f0),
            Dropout(0.25),
			x -> reshape(x,128,:),
			x->reshape(x, 7 * 7 * 128, :),
            Dense(7 * 7 * 128, 1),
			x -> leakyrelu.(x,0.4f0))
end

function Generator(hparams)
    return Chain(
			Dense(hparams.latent_dim, 7*7*256),
			x -> leakyrelu.(x,0.3f0),
			Dropout(0.3),
			BatchNorm(7 * 7 * 256, relu),
            x->reshape(x, 7, 7, 256, :),
			ConvTranspose((5, 5), 256 => 128; stride = 1, pad = 2),
            BatchNorm(128, relu),
            ConvTranspose((4, 4), 128 => 64; stride = 2, pad = 1),
            BatchNorm(64, relu),
            ConvTranspose((4, 4), 64 => 1, tanh; stride = 2, pad = 1),
			x->transpose.(x)
			)
end

function Observer()
    return Chain(
			x -> reshape(x,4,:),
			Dense(4,256),
			x -> leakyrelu.(x,0.15f0),
			Dropout(0.25),
			BatchNorm(256, relu),
			x -> reshape(x, 4,4,16,:),
			Conv((3, 3), 16 => 64; stride = 1, pad = 1),
			x -> leakyrelu.(x,0.2f0),
			Dropout(0.2),
			#Conv((3, 3), 64 => 128; stride = 1, pad = 1),
			#x -> leakyrelu.(x,0.2f0),
			#Dropout(0.2),
			#x -> reshape(x,128,:),
			#Dense(128, 256),
			#x -> leakyrelu.(x,0.2f0),
			#Dropout(0.3),
			#Dense(256,64),
			#x -> leakyrelu.(x,0.3f0),
			#Dropout(0.2),
			x -> reshape(x,64,:),
			Dense(64, 1),
			x -> leakyrelu.(x,0.2f0))
end

# Discriminator loss
function discriminator_loss(real_output, fake_output)
	real_loss = mean(logitbinarycrossentropy.(real_output,
											  ones(size(real_output)) |> gpu))
	fake_loss = mean(logitbinarycrossentropy.(fake_output,
											  zeros(size(fake_output)) |> gpu))
	return real_loss + fake_loss
end

# Observer loss
function observer_loss(obs_real, obs_fake, control_factor)
	loss = 0.0
	obs_real = obs_real # |> cpu
	obs_fake = obs_fake # |> cpu

	loss_trues = log(mean(logitbinarycrossentropy.(
						obs_real,cu(zeros(size(obs_real))))))
	loss_falses = log(mean(logitbinarycrossentropy.(
						obs_fake,cu(ones(size(obs_fake))))))

	loss = control_factor * abs(loss_trues+loss_falses)
	return loss |> gpu
end

# Generator loss
function generator_loss(fake_output, real_output) 
	loss_fake = mean(
				logitbinarycrossentropy.(fake_output,
				ones(size(fake_output)) |> gpu))
	loss_real = mean(
				logitbinarycrossentropy.(real_output,
				zeros(size(real_output)) |> gpu))
	loss = loss_real + loss_fake
	return loss |> gpu
end
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
	obs_ones = cu(ones(size(false_observation))) |> gpu
	obs_zeros = cu(zeros(size(real_observation))) |> gpu

	diff_1 = log(mean(logitbinarycrossentropy.(real_observation, obs_zeros)))
	diff_2 = log(mean(logitbinarycrossentropy.(false_observation, obs_ones)))
	observer_distance = (diff_1 + diff_2)

	return discriminator_distance < abs(control_factor * observer_distance)
end

function make_observations!(gen,dscr,obs, x, opt_dscr,opt_gen,opt_obs,
							hparams::HyperParams,control_factor::Float64,
							rng::MersenneTwister, controller::Bool=true)
	counter_in_loop = 0
	loss_obs = 0.0

	rr = hparams.reduction_rate
	lc = hparams.loop_condition
	while controller
		# Now we will train the observer
		noise = randn!(
				similar(x, (hparams.latent_dim, hparams.batch_size))) |> gpu
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
			observer_loss(real_obs, fake_obs, control_factor)
		end
		grad_obs = back_obs(1f0)
		update!(opt_obs, ps_obs, grad_obs)

		# now recompute the discriminator loss
		# with the K-Lipschitz condition
		# where K = control_factor
		noise = randn!(
				similar(x,(hparams.latent_dim, hparams.batch_size))) |> gpu
		fake_input = gen(noise)
		ps_dscr = Flux.params(dscr)
		#Taking gradient
		loss_dscr, back_dscr = Flux.pullback(ps_dscr) do
			discriminator_loss(dscr(fake_input), dscr(fake_input))
		end
		grad_dscr = back_dscr(1f0)
		update!(opt_dscr, ps_dscr, grad_dscr)

		# apply the Lipschitz condition
		real_preimage =  dscr(x) |> gpu
		fake_preimage =  dscr(gen(noise)) |> gpu
		controller = !lipschitz_bound(control_factor, obs,
									 real_preimage,fake_preimage)

		# If we are stayin in the loo[ for too long,
		# adjust the control_factor
		counter_in_loop += 1
		
		if counter_in_loop > hparams.loop_condition * hparams.multiplier_lc
			controller = false
			# even the Lipschitz condition did meet the criteria,
			# strengthen the condition for the next step
			# do not strengthen the condition far too small though
			if hparams.loop_condition > hparams.reduction_limit
				hparams.reduction_rate = rr*(1.0 + randn(1)[1])
			end
		elseif counter_in_loop > hparams.loop_condition
			# train the generator when the loop_condition is violated
			loss_gen = train_generator!(gen,dscr,x,opt_gen, hparams)
			control_factor = control_factor * rr
		end
	end
	return loss_obs, control_factor
end

# Train discriminator
function train_discriminator!(gen, dscr, x, opt_dscr, hparams::HyperParams)
	# initialize
	loss_obs = 0.0
	noise = randn!(similar(x, (hparams.latent_dim, hparams.batch_size))) |>gpu
	fake_input = gen(noise)
	ps = Flux.params(dscr)
	#Taking gradient
	loss, back = Flux.pullback(ps) do
		discriminator_loss(dscr(cu(x)), dscr(fake_input))
	end
	grad = back(1f0)
	update!(opt_dscr, ps, grad)

	return loss
end


# Train Generator (and the observer...)
function train_generator!(gen, dscr, x, opt_gen, hparams::HyperParams)
	noise = cu(randn!(
			similar(x, (hparams.latent_dim, hparams.batch_size)))) |> gpu

	# Taking gradient
		control_factor = cu(rand(1)[1]) |>gpu
		loss = 0.0
		noise = cu(randn!(
				similar(x, (hparams.latent_dim, hparams.batch_size)))) |> gpu

		ps = Flux.params(gen) |> gpu
		dscr_x = dscr(cu(x))
		loss, back = Flux.pullback(ps) do
			generator_loss(dscr(gen(noise)), dscr_x)
		end
		grad = back(1f0)
		update!(opt_gen, ps, grad)

		return loss
end

function lipschitz_switch(gen,dscr,obs,control_factor, x, hparams::HyperParams)
	noise = cu(randn!(
			similar(x, (hparams.latent_dim, hparams.batch_size)))) |> gpu
	real_preimage = dscr(cu(x)) |> gpu
	fake_preimage = dscr(gen(noise)) |> gpu

	switch = lipschitz_bound(control_factor,obs, real_preimage, fake_preimage)
	return switch
end

function train(; kws...)
	# Model Parameters
	hparams = HyperParams(; kws...)
	
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
	# Load MNIST dataset
	images, _ = MLDatasets.MNIST.traindata(Float32)
	# Normalize to [-1, 1]
	image_tensor = reshape(@.(2f0 * images - 1f0), 28, 28, 1, :)
	# Partition into batches
	data = [image_tensor[:, :, :, r] for r in partition(1:60000, hparams.batch_size)]

	fixed_noise = [cu(randn(hparams.latent_dim, 1)) |> gpu
					for _=1:hparams.output_x*hparams.output_y]

	# Discriminator
	dscr = Discriminator() |> gpu
	# Generator
	gen =  Generator(hparams) |> gpu
	# Observer
	obs = Observer() |> gpu

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
		for x in ProgressBar(data)
			# add noise on x every hparams.noise_addition steps
			if train_steps % hparams.noise_addition == 0
				x = x .* randn(size(x))
			elseif train_steps % hparams.noise_addition == 2
				x = x .+ randn(size(x))
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
				# Save generated fake image
				output_image = create_output_image(gen, fixed_noise, hparams)
				output_filename = "output/step_$(train_steps).png"
				println(output_filename)
				save(output_filename, output_image)
			end
			train_steps += 1
		end
	end
	close(io) # closing the log file
	output_image = create_output_image(gen, fixed_noise, hparams)
	save(@sprintf("output/last_step_%06d.png", train_steps), output_image)
end

#
#	Run the train function
#
cd(@__DIR__)
train()
