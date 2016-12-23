require 'stn'
require 'nn';
npy4th=require 'npy4th'
require 'cudnn';

local num_ctrl_pts=16;

spanet=nn.Sequential()

local concat=nn.ConcatTable()

-- first branch is there to transpose inputs to BHWD, for the bilinear sampler
tranet=nn.Sequential()
tranet:add(nn.Identity())
tranet:add(nn.Transpose({2,3},{3,4}))

-- second branch is the localization network
local locnet = nn.Sequential()
locnet:add(cudnn.SpatialMaxPooling(2,2,2,2))
locnet:add(cudnn.SpatialConvolution(1,20,5,5))
locnet:add(cudnn.ReLU(true))
locnet:add(cudnn.SpatialMaxPooling(2,2,2,2))
locnet:add(cudnn.SpatialConvolution(20,20,5,5))
locnet:add(cudnn.ReLU(true))
locnet:add(nn.View(20*2*2))
locnet:add(nn.Linear(20*2*2,20))
locnet:add(cudnn.ReLU(true))

-- we initialize the output layer so it gives the identity transform
local outLayer = nn.Linear(20,2*num_ctrl_pts)
outLayer.weight:fill(0)
local bias = torch.FloatTensor(2*num_ctrl_pts):fill(0)
outLayer.bias:copy(bias)
locnet:add(outLayer)

-- there we generate the grids
-- locnet:add(nn.View(2*num_ctrl_pts))
locnet:add(nn.TPSGridGeneratorBHWD(num_ctrl_pts,32,32));


-- local outLayer = nn.Linear(20,6)
-- outLayer.weight:fill(0)
-- local bias = torch.FloatTensor(6):fill(0)
-- bias[1]=1
-- bias[5]=1
-- outLayer.bias:copy(bias)
-- locnet:add(outLayer)
-- locnet:add(nn.View(2,3))
-- locnet:add(nn.AffineGridGeneratorBHWD(32,32));

-- print (tps.batchGrid);
-- print ('tps.baseGrid',tps.baseGrid:size())
-- print ('tps.batchGrid',tps.batchGrid:size())
-- print ('tps.right_mat',tps.right_mat:size())
-- print ('tps.L_inv',tps.L_inv:size())
-- print ('tps.source_points',tps.source_points:size())
-- print ('tps.A',tps.A:size())


-- npy4th.savenpy('batchGrid.npy',tps.batchGrid);
-- npy4th.savenpy('baseGrid.npy',tps.baseGrid)
-- npy4th.savenpy('batchGrid.npy',tps.batchGrid)
-- npy4th.savenpy('right_mat.npy',tps.right_mat)
-- npy4th.savenpy('L_inv.npy',tps.L_inv)
-- npy4th.savenpy('source_points.npy',tps.source_points)
-- npy4th.savenpy('A.npy',tps.A)



-- we need a table input for the bilinear sampler, so we use concattable
concat:add(tranet)
concat:add(locnet)

spanet:add(concat)
spanet:add(nn.BilinearSamplerBHWD())

-- and we transpose back to standard BDHW format for subsequent processing by nn modules
spanet:add(nn.Transpose({3,4},{2,3}))
