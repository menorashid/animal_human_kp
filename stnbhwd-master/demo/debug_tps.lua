require 'cunn'
require 'cudnn'
require 'nn';
require 'image'
require 'optim'
require 'stn'
npy4th=require 'npy4th';
paths.dofile('Optim.lua')



function getLossD(pred_output,gt_output)
	local lossD=pred_output-gt_output
	lossD=torch.mul(lossD,2);
	-- print (torch.mean(lossD));
	return lossD;
end

function getLoss(pred_output,gt_output)
	local loss=torch.pow(pred_output-gt_output,2);
	loss=torch.mean(loss);
	return loss;
end

local fevalScore = function(x)
    if x ~= parameters then
	    parameters:copy(x)
    end
    
    local batch_inputs = data:clone();
    local batch_targets = grids:clone();
    
    gradParameters:zero()
    if dest_flag then
		batch_inputs = dest_offsets:clone();
	end

    local outputs=locnet:forward(batch_inputs);
    local dloss = getLossD(outputs,batch_targets);
    local loss = getLoss(outputs,batch_targets);

    locnet:backward(batch_inputs, dloss)
    -- local x=locnet:get(#locnet).gradInput:clone();
    -- x=x:view(x:nElement())
    -- x=torch.sqrt(torch.sum(torch.pow(x,2)));
    -- print (x)
    -- local x=locnet:get(#locnet-1).gradInput:clone();
    -- x=x:view(x:nElement())
    -- x=torch.sqrt(torch.sum(torch.pow(x,2)));
    -- print (x)
    
    -- print ('dloss[1][1]');
    -- print (dloss[1][1]);
    -- print ('outputs[1][1]');
    -- print (outputs[1][1]);
    -- print ('grids[1][1]');
    -- print (grids[1][1]);
    
    -- print (locnet:get(#locnet).gradInput[1])
    -- print (outputs:size());
    -- print (dest_offsets[1]);

    
    return loss, gradParameters;

    -- local batch_outputs_mid = net:get(1):forward{batch_inputs,batch_inputs_flow}
    -- local batch_outputs = net:get(2):get(1):forward(batch_outputs_mid)

    -- local batch_loss = getLossScore(batch_outputs, batch_targets)
    -- local dloss_doutput = getLossScoreD(batch_outputs, batch_targets)
    -- local gradInputs=net:get(2):get(1):backward(batch_outputs_mid, dloss_doutput)
    -- net:get(1):backward(batch_inputs, gradInputs)
    -- return batch_loss, gradParameters
end

-- paths.dofile('distort_mnist.lua')
-- datasetTrain, datasetVal = createDatasetsDistorted()

-- local trainFileName = 'mnist.t7/test_32x32.t7'
-- local train = torch.load(trainFileName, 'ascii')
-- train.data = train.data:float()
-- train.labels = train.labels:float();

local out_dir='/home/SSD3/maheen-data/mnist_subset'
print (out_dir);
paths.mkdir(out_dir);

-- for i=1,10 do
	-- print (train.labels[i]);
-- 	local im=train.data[i][1];
-- 	local out_file=paths.concat(out_dir,i..'.jpg');
-- 	image.save(out_file,im);
-- end

data=torch.zeros(10,1,32,32);
for i=1,10 do
	local file_curr=paths.concat(out_dir,i..'.jpg');
	local im=image.load(file_curr);
	-- print (im:size());
	-- im=im:view(1,im:size(1),im:size(2));
	-- print (torch.min(im),torch.max(im));	
	data[i]=im;
end

-- grids=npy4th.loadnpy('dummy_grids.npy');
grids=npy4th.loadnpy('out_grids_true_2.npy');
-- grids=npy4th.loadnpy('outputs_true.npy')
-- dest_offsets=npy4th.loadnpy('dummy_dest_trans.npy')*-1;
dest_offsets=npy4th.loadnpy('dummy_dest_rand.npy')*-1;

-- print (dest_offsets:size())
-- print (dest_offsets[1]);
-- print (grids:size());

num_ctrl_pts=16;

data=data:cuda();
grids=grids:cuda();
dest_offsets=dest_offsets:cuda();

dest_flag=false;
max_iter=60;

-- dest_flag=true;
-- max_iter=1;

print (data:size())

	
locnet = nn.Sequential()
locnet:add(cudnn.SpatialMaxPooling(2,2,2,2))
locnet:add(cudnn.SpatialConvolution(1,20,5,5))
locnet:add(cudnn.ReLU(true))
locnet:add(cudnn.SpatialMaxPooling(2,2,2,2))
locnet:add(cudnn.SpatialConvolution(20,20,5,5))
locnet:add(cudnn.ReLU(true))
locnet:add(nn.View(20*2*2))
locnet:add(nn.Linear(20*2*2,20))
locnet:add(cudnn.ReLU(true))

-- -- we initialize the output layer so it gives the identity transform
-- local outLayer = nn.Linear(20,6)
-- outLayer.weight:fill(0)
-- local bias = torch.FloatTensor(6):fill(0)
-- bias[1]=1
-- bias[5]=1
-- outLayer.bias:copy(bias)
-- locnet:add(outLayer)

-- -- there we generate the grids
-- locnet:add(nn.View(2,3))
-- locnet:add(nn.AffineGridGeneratorBHWD(32,32))

-- we initialize the output layer so it gives the identity transform
local outLayer = nn.Linear(20,2*num_ctrl_pts)
outLayer.weight:fill(0)
local bias = torch.FloatTensor(2*num_ctrl_pts):fill(0)
outLayer.bias:copy(bias)
locnet:add(outLayer)
-- there we generate the grids
locnet:add(nn.TPSGridGeneratorBHWD(num_ctrl_pts,32,32));

if dest_flag then
	locnet = nn.Sequential();
	locnet:add(nn.TPSGridGeneratorBHWD(num_ctrl_pts,32,32));
end

-- print (locnet:get(1).L:size())
-- print (locnet:get(1).L)

locnet:cuda();
print (locnet);

parameters, gradParameters = locnet:getParameters()

optimState = {learningRate = 0.001, momentum = 0.9, weightDecay = 5e-4}
optimMethod = optim.sgd

for i=1,max_iter do

	local _, minibatch_loss = optimMethod(fevalScore,parameters, optimState)
	-- print (minibatch_loss[1]:size());
	-- print (i,torch.mean(minibatch_loss[1]))
	print(string.format("minibatches processed: %6s, loss = %6.6f", i, minibatch_loss[1]));
    
end






local out_grids;
if dest_flag then
	out_grids=locnet:forward(dest_offsets:clone());
	npy4th.savenpy('out_grids_true_2.npy',out_grids);
else
	out_grids=locnet:forward(data:clone());	
end


-- print (out_grids:size());
-- print (data:size());

k=nn.Transpose({2,3},{3,4}):cuda()

data=k:forward(data);

-- print (data:size());

x=nn.BilinearSamplerBHWD()
x:forward({data:double(),out_grids:double()});
-- print (x.output:size());


for i=1,10 do
    local im=x.output[i]:view(1,32,32);
    local out_file_curr=paths.concat(out_dir,i..'_t.jpg');
    image.save(out_file_curr,im);
    print (out_file_curr);
end

-- tps=nn.TPSGridGeneratorBHWD(num_ctrl_pts,32,32):cuda();






-- print (tps);
-- input=dest_offsets*-1;
-- input=input:cuda();

-- for i=1,10 do
	
-- 	tps:forward(input:clone());
-- 	out_grids=tps.output;
-- 	out_grids=out_grids:double();

-- 	print (i);

-- 	-- print ('out_grids',out_grids:size(),out_grids:type())
-- 	-- print ('grid',grids:size(),grids:type())

-- 	print (torch.max(torch.abs(out_grids-grids)));
-- 	print (torch.mean(getLossD(out_grids,grids)));
-- 	print (getLoss(out_grids,grids));

-- end





-- x=nn.BilinearSamplerBHWD()
-- x:forward({data,out_grids});
-- print (x.output:size());


-- for i=1,10 do
--     local im=x.output[i]:view(1,32,32);
--     local out_file_curr=paths.concat(out_dir,i..'_t.jpg');
--     image.save(out_file_curr,im);
--     print (out_file_curr);
-- end



-- local im=train.data[1]
-- print (torch.min(im),torch.max(im));


-- -- out_file=
-- print (train.data:size(),train.data:type())
-- print (train.labels:size(),train.labels:type())

