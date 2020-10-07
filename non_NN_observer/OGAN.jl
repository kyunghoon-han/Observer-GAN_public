using Base.Iterators: partition
# Flux stuff
using Flux, CUDA
import Flux.Losses: mse
using Flux.Optimise: Momentum, update!
# Image stuff
using Images, ImageIO, ImageMagick
# Plots
using Plots
# Datasets
using MLDatasets
using Statistics
using Parameters
# Other stuff
using LinearAlgebra
using Random
using ProgressBars
import Distributions: Uniform

using Suppressor

@with_kw mutable struct HPARAMS
	batch_size::Int = 64
	epochs::Int = 200
	# feedback frequency
	verbose_freq::Int = 500
	# learning rate
	lr::Float64 = 0.001
	# tolerance on ratios
	tolerance::Float64 = 0.001
	threshold::Float64 = 0.3
	max_loop::Int = 20
end

# load full training set
#train_x, train_y = MNIST.traindata()
train_x, train_y = FashionMNIST.traindata()
# load full test set
#test_x,  test_y  = MNIST.testdata()
test_x,  test_y  = FashionMNIST.testdata()
function model1(hparams)
	return Chain(
		x -> reshape(x,28*28,:),
		Dense(28*28,512),
		x -> leakyrelu.(x,0.3f0),
		Dropout(0.2),
		Dense(512, 32),
		x -> leakyrelu.(x, 0.3f0),
		Dropout(0.2),
		Dense(32,1),
		x -> leakyrelu.(x, 0.2f0)
		)
end

function model2(hparams)
	return Chain(
		x -> reshape(x,28*28,:),
		Dense(28*28,512),
		x -> leakyrelu.(x,0.3f0),
		Dropout(0.2),
		Dense(512, 32),
		x -> leakyrelu.(x, 0.3f0),
		Dropout(0.2),
		Dense(32,1),
		x -> leakyrelu.(x, 0.2f0)
	
		)
end

#=
function model(hparams)
	return Chain(
		Conv((2,2), 1 => 4; stride=1, pad=2),
		x -> leakyrelu.(x, 0.2f0),
		Dropout(0.2),
		Conv((2,2), 4=>16; stride=1, pad=2),
		x -> leakyrelu.(x, 0.2f0),
		Dropout(0.2),
		Conv((3,3), 16=>8; stride=1, pad=1),
		x -> leakyrelu.(x, 0.2f0),
		Dropout(0.2),
		Conv((3,3), 8=>4; stride=1, pad=1),
		x -> leakyrelu.(x,0.2f0),
		Dropout(0.2),
		Conv((2,2), 4=>1; stride=1, pad=1),
		x -> leakyrelu.(x, 0.2f0),
		Dropout(0.2),
		x -> reshape(x,100,:),
		Dense(100,hparams.batch_size),
		x -> leakyrelu.(x,0.2f0),
		x -> sum(x,dims=2),
		x -> reshape(x,hparams.batch_size),
		Dense(hparams.batch_size,hparams.batch_size),
		x -> sigmoid.(x)
	)
end
=#

function discrete_observation(from_nn,target,hparams)
    max_target = maximum(target)
    output = round.(from_nn .* max_target) .- max_target*0.5

    ratio_positive = length([i for i in output if i>0]) / length(output)
    ratio_negative = length([i for i in output if i<0]) / length(output)
	target_ratio_p = length([i for i in target if i>0]) / length(output)
	target_ratio_n = length([i for i in target if i<0]) / length(target)

    ratio = ratio_positive - ratio_negative
	target_ratio = target_ratio_p - target_ratio_n

	if ratio < hparams.tolerance
		return 1.0
	else
		return (target_ratio-ratio)/(target_ratio + ratio)
	end
end

function criterion(output,target)
    return Flux.mse(output,target)
end

function train_model(x,y,model, optim, switch, hparams)
    ps = Flux.params(model)
    a = model(x)
    y = reshape(y,size(a))
    loss = 0.0

    if switch
        factor = discrete_observation(a,y,hparams)
		counter = 0
		while factor > hparams.threshold
			gs = gradient(ps) do
				loss = criterion(model(x),y)
			end
			update!(optim,ps,gs)
			a = model(x)
			factor = discrete_observation(a,y,hparams)
			counter += 1
			if counter > hparams.max_loop
				return loss
			end
		end
    else
        factor = 1.0
    end

    gs = gradient(ps) do
        loss = criterion(model(x),y)
    end
    update!(optim, ps, gs)

    return loss
end

function less_than(a,b)
	if a < b
		return 1.0
	else
		return 0.0
	end
end

accuracy(x,y,m) = mean(onecold(cpu(m(x)), 1:10) .== onecold(cpu(y), 1:10))

function test(model_interest, x, y, hparams)
    x = reshape(@.(2f0 .* x .- 1f0), 28, 28, :)
    y = reshape(y ./ 9.0, 10000)
    data_in = [x[:, :, r] for r in partition(1:10000, hparams.batch_size)];
    targets = [y[r] for r in partition(1:10000, hparams.batch_size)];
    tolerance = 0.5
	counter = 0
    max_acc = 0.0
    for x in data_in
        counter += 1
		val = targets[counter]
		pred = abs.(model_interest(x)) .* 9.0
		val = abs.(reshape(val,size(pred))) .* 9.0
		acc = Statistics.mean([less_than(i,tolerance) for i in abs.(pred .- val)])
		if max_acc < acc
			max_acc = acc
		end
		if counter > 154
			return 100.0*max_acc
		end
	end
	
	return 100.0*max_acc
end

function train(; kws...)
    hparams = HPARAMS(; kws...)
    # normalize the data to -1 to 1
	image_tensor = reshape(@.(2f0 .* train_x .- 1f0), 28, 28,:);
	answer_tensor = reshape(train_y/9.0, 60000)
	# parition the above tensor to batches
	data_x = [image_tensor[:, :, r] for r in partition(1:60000, hparams.batch_size)];
	data_y = [answer_tensor[r] for r in partition(1:60000, hparams.batch_size)];
	# models
    without_observation = model1(hparams) |> gpu
    with_observation = model2(hparams) |> gpu
    # optimisers
    no_o_optim = Momentum(hparams.lr,0.9)
    o_optim = Momentum(hparams.lr,0.9)
	train_step = 0

	if ispath("./logs")
		rm("./logs",recursive=true)
		mkpath("./logs")
	else
		mkpath("./logs")
	end

	a = "./logs/logs.csv"
	first_line = "Step, Accuracy with an observer, Accuracy without an observer\n"
	io = open(a, "a")
   for epoch in 1:hparams.epochs
        counter = 0
        for x in ProgressBar(data_x)
            counter = counter + 1 # this will count the entries of data_y
            x = cu(x)
            y = cu(data_y[counter])
            #if counter == length(data_y)-1
            #    break
            #end
            # now pass the data through the networks
        without_loss = train_model(x,y,without_observation,no_o_optim,false,hparams)
        with_loss = train_model(x,y,with_observation, o_optim,true,hparams)

			# iterate the train_step
			train_step += 1

            # test
            if train_step % hparams.verbose_freq == 0
                without = test(without_observation, cu(test_x),cu(test_y), hparams)
                with = test(with_observation, cu(test_x),cu(test_y), hparams)
				println()
				#println("At $(epoch) epoch , $(counter) step.")
                #println("Without observations, the loss is : ", without_loss)
                #println("With observations, the loss is : ", with_loss)
                #println("Without observations, we have an accuracy of $(without)%")
                #println("With observations, we have an accuracy of $(with)%")
				printer = "$(train_step), $(with)%, $(without)%\n"
				write(io,printer)
            end # end of test
        end # end of inner for-loop
    end # end of epoch for-loop
	close(io)
end

@suppress_err begin
    train()
end
