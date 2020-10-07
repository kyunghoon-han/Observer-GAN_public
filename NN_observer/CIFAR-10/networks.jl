using Flux
include("hparams.jl")

function Discriminator()
	return Chain(
			x -> reshape(x,4,:),
			Dense(4,128),
			x -> leakyrelu.(x,0.3f0),
			Dropout(0.2),
			Dense(128, 64),
			x -> leakyrelu.(x, 0.2f0),
			Dropout(0.2),
			x -> reshape(x,2,2,16,:),
			Conv((4, 4), 16 => 8; stride = 1, pad = 1),
			x->leakyrelu.(x,0.2f0),
			Dropout(0.2),
			Conv((3,3), 8=>4; stride = 1, pad = 1),
			x->leakyrelu.(x,0.2f0),
			x->reshape(x,3*4,:),
			Dense(3*4, 1),
			x->leakyrelu.(x,0.2f0),
			)
end

function Generator(hparams)
     return Chain(
			x -> reshape(x,4,4,12,:),
			Conv((2,2), 12=>hparams.latent_dim;stride=1,pad=1),
			x -> leakyrelu.(x,0.2f0),
			Dropout(0.2),
			x -> reshape(x,hparams.latent_dim,:),
            Dense(hparams.latent_dim, 3*3*64),
			x -> leakyrelu.(x,0.3f0),
            Dropout(0.3),
            BatchNorm(5 * 5 * 64, relu),
            x->reshape(x, 5, 5, 64, :),
            ConvTranspose((3, 3), 64 => 16; stride = 1, pad = 2),
            BatchNorm(16, relu),
            ConvTranspose((3, 3), 16 => 8; stride = 2, pad = 1),
            BatchNorm(8, relu),
            ConvTranspose((3, 3), 8 => 3, tanh; stride = 2, pad = 1),
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
			 x -> reshape(x,64,:),
             Dense(64, 1),
             x -> leakyrelu.(x,0.2f0))
end

