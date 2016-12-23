local TGG, parent = torch.class('nn.TPSGridGeneratorBHWD', 'nn.Module')

--[[
   AffineGridGeneratorBHWD(height, width) :
   AffineGridGeneratorBHWD:updateOutput(transformMatrix)
   AffineGridGeneratorBHWD:updateGradInput(transformMatrix, gradGrids)

   AffineGridGeneratorBHWD will take 2x3 an affine image transform matrix (homogeneous 
   coordinates) as input, and output a grid, in normalized coordinates* that, once used
   with the Bilinear Sampler, will result in an affine transform.

   AffineGridGenerator 
   - takes (B,2,3)-shaped transform matrices as input (B=batch).
   - outputs a grid in BHWD layout, that can be used directly with BilinearSamplerBHWD
   - initialization of the previous layer should biased towards the identity transform :
      | 1  0  0 |
      | 0  1  0 |

   *: normalized coordinates [-1,1] correspond to the boundaries of the input image. 
]]

function TGG:__init(num_ctrl_pts,out_height,out_width)
   parent.__init(self)


   assert(torch.sqrt(num_ctrl_pts) %1 == 0)
   self.num_ctrl_pts = num_ctrl_pts
   height =  torch.sqrt(num_ctrl_pts);
   width =   torch.sqrt(num_ctrl_pts);
   self.height =height;
   self.width =width;

   self.out_height=out_height;
   self.out_width=out_width;

   self.baseGrid,self.batchGrid, self.right_mat, self.L_inv, self.source_points,self.orig_grid=
                     self:initialize_tps(self.height,self.width,self.out_height,self.out_width);

   self.A = self.L_inv[{{},{4,self.L_inv:size(2)}}]:t()*self.right_mat;
  
   

end

local function _U_func(x1,y1,x2,y2)

    local ans;
    if x1 == x2 and y1 == y2 then
        ans= 0;
    else
      ans = (x2 - x1)^2 + (y2 - y1)^2
      ans = ans * torch.log(ans) 
    end    
    
    if 'inf'==''..ans or 'nan'==''..ans then
      ans=0;
    end

    return ans;
end

function TGG:initialize_tps()
   local num_control_points=self.num_ctrl_pts;
   local height =self.height;
   local width = self.width;
   local out_height=self.out_height;
   local out_width=self.out_width;
   local baseGrid = torch.zeros(height, width, 3)
   
   for i=1,height do
     baseGrid:select(3,1):select(1,i):fill(-1 + (i-1)/(height-1) * 2)
   end
   for j=1,width do
     baseGrid:select(3,2):select(2,j):fill(-1 + (j-1)/(width-1) * 2)
   end
   baseGrid:select(3,3):fill(1)
   local batchGrid = torch.Tensor(1, height, width, 3):copy(baseGrid)
   
   local num_equations = num_control_points + 3
   local L = torch.zeros(num_equations, num_equations)
   
   local baseGrid_t=baseGrid:resize(num_control_points,3);
   baseGrid_t=baseGrid_t[{{},{1,2}}]
   L[{{2,3},{4,L:size(2)}}]=baseGrid_t:t()
   L[{1, {4,L:size(2)}}]=1
   L[{{4,L:size(2)},{2,3}}]=baseGrid_t
   L[{{4,L:size(2)},1}]=1
   for point_1=1,num_control_points do
     for point_2=point_1,num_control_points do
       L[{point_1+3,point_2+3}]=_U_func(baseGrid_t[{point_1,1}],baseGrid_t[{point_1,2}],baseGrid_t[{point_2,1}],baseGrid_t[{point_2,2}]);
       if point_1 ~= point_2 then
         L[{point_2+3,point_1+3}]=L[{point_1+3,point_2+3}]
       end
     end
   end
   self.L=L;

   local L_inv=torch.inverse(L);

   local orig_grid=torch.zeros(out_height,out_width,3);
   for i=1,out_height do
     orig_grid:select(3,1):select(1,i):fill(-1 + (i-1)/(out_height-1) * 2)
   end
   for j=1,out_width do
     orig_grid:select(3,2):select(2,j):fill(-1 + (j-1)/(out_width-1) * 2)
   end
   orig_grid:select(3,3):fill(1)

   orig_grid=orig_grid:resize(out_height*out_width,3);
   orig_grid=orig_grid[{{},{1,2}}]:t();
   local source_pts=baseGrid_t:t();

   local distances =torch.zeros(source_pts:size(2),orig_grid:size(2));
   for ps_idx=1,source_pts:size(2) do
     for po_idx=1,orig_grid:size(2) do
       distances[{ps_idx,po_idx}]=_U_func(source_pts[{1,ps_idx}],source_pts[{2,ps_idx}],orig_grid[{1,po_idx}],orig_grid[{2,po_idx}]);
     end
   end

   local upper_array=torch.ones(1,orig_grid:size(2));
   upper_array=torch.cat({upper_array,orig_grid},1);
   local right_mat=torch.cat({upper_array,distances},1);

   return baseGrid, batchGrid, right_mat, L_inv, source_pts, orig_grid;
end
    
local function addOuterDim(t)
   local sizes = t:size()
   local newsizes = torch.LongStorage(sizes:size()+1)
   newsizes[1]=1
   for i=1,sizes:size() do
      newsizes[i+1]=sizes[i]
   end
   return t:view(newsizes)
end


function TGG:updateOutput(_destOffsets)
   local dest_offsets
   local A=self.A:clone();
   local out_width=self.out_width;
   local out_height=self.out_height;
   local source_points = self.source_points:clone();
   local L_inv_t=self.L_inv[{{},{4,self.L_inv:size(2)}}]:t():clone();

   if _destOffsets:nDimension()==1 then
      dest_offsets = addOuterDim(_destOffsets)
   else
      dest_offsets = _destOffsets:clone();
   end

  local num_control_points = source_points:size(2)
  local num_batch=dest_offsets:size(1);
  dest_offsets=dest_offsets:resize(torch.LongStorage{num_batch,num_control_points,2}):transpose(2,3);
  local dest_points=torch.zeros(dest_offsets:size())
  
  local s_pts_rep=source_points:resize(1,source_points:size(1),source_points:size(2));
  
  s_pts_rep=s_pts_rep:repeatTensor(num_batch,1,1);
 
  local dest_points=s_pts_rep+dest_offsets;
 
  local transformed_points=torch.bmm(dest_points,torch.repeatTensor(A,dest_points:size(1),1,1));
  
  self.coefficients = torch.bmm(dest_points,torch.repeatTensor(L_inv_t,dest_points:size(1),1,1));

  local x_t_all=torch.Tensor(torch.LongStorage{transformed_points:size(1),out_height,out_width}):type(transformed_points:type());
  local y_t_all=torch.Tensor(torch.LongStorage{transformed_points:size(1),out_height,out_width}):type(transformed_points:type());
  
  for idx=1,transformed_points:size(1) do
    x_t_all[idx]=transformed_points[idx][1]:resize(out_height,out_width);
    y_t_all[idx]=transformed_points[idx][2]:resize(out_height,out_width);
  end

  self.output=torch.cat(x_t_all,y_t_all,4);

  return self.output;

end


local function getAMatrix(baseGrid_t,orig_grid)
   local num_control_points=baseGrid_t:size(2);
   
   baseGrid_t = baseGrid_t:t();
   
   local num_equations = num_control_points + 3
   local L = torch.zeros(num_equations, num_equations):type(baseGrid_t:type());

   
   L[{{2,3},{4,L:size(2)}}]=baseGrid_t:t();
   L[{1, {4,L:size(2)}}]=1
   L[{{4,L:size(2)},{2,3}}]=baseGrid_t;
   L[{{4,L:size(2)},1}]=1
   for point_1=1,num_control_points do
     for point_2=point_1,num_control_points do
         L[{point_1+3,point_2+3}]=_U_func(baseGrid_t[{point_1,1}],baseGrid_t[{point_1,2}],baseGrid_t[{point_2,1}],baseGrid_t[{point_2,2}]);
         if point_1 ~= point_2 then
            L[{point_2+3,point_1+3}]=L[{point_1+3,point_2+3}]
         end
     end
   end
   local L_inv=torch.inverse(L);
   if (orig_grid:type()~=baseGrid_t:type()) then
      orig_grid=orig_grid:type(baseGrid_t:type());
   end
  
   local source_pts=baseGrid_t:t();
   
   local num_orig=orig_grid:size(2);
   local num_sp=source_pts:size(2);
   local s_p_x=torch.repeatTensor(source_pts:select(1,1):clone():view(source_pts:size(2),1),1,num_orig);
   local s_p_y=torch.repeatTensor(source_pts:select(1,2):clone():view(source_pts:size(2),1),1,num_orig);
   
   local o_g_x=torch.repeatTensor(orig_grid:select(1,1):clone(),num_sp,1);
   local o_g_y=torch.repeatTensor(orig_grid:select(1,2):clone(),num_sp,1);

   local distances_new=torch.pow(s_p_x-o_g_x,2)+torch.pow(s_p_y-o_g_y,2);
   distances_new= distances_new:cmul(torch.log(distances_new));
   distances_new[distances_new:ne(distances_new)]=0;
   distances_new[distances_new:eq(3/0)]=0;
   local upper_array=torch.ones(1,orig_grid:size(2)):type(baseGrid_t:type());
   upper_array=torch.cat({upper_array,orig_grid},1);
   local right_mat=torch.cat({upper_array,distances_new},1);
   local A= L_inv[{{},{4,L_inv:size(2)}}]:t()*right_mat;
   return A;
end

function TGG:getGTOutput(human_keypoints,horse_keypoints)
   -- local source_pts = self.source_points:clone();
   local orig_grid = self.orig_grid:clone();
   local gt_output=torch.zeros(#human_keypoints,self.out_height,self.out_width,2):type(human_keypoints[1]:type());
   
   for idx_batch=1,#human_keypoints do
      -- print (human_keypoints[idx_batch]:size())
      local A=getAMatrix(human_keypoints[idx_batch],orig_grid);
      local transformed_points=horse_keypoints[idx_batch]*A;
      local x_t_all=transformed_points[1]:resize(self.out_height,self.out_width);
      local y_t_all=transformed_points[2]:resize(self.out_height,self.out_width);
      gt_output[{idx_batch,{},{},1}]=x_t_all;
      gt_output[{idx_batch,{},{},2}]=y_t_all;
   end
   return gt_output;
   
end

function TGG:updateOutputForSize(out_size)
   local out_height=out_size[1];
   local out_width=out_size[2];
   local source_pts=self.source_points;
   
   local orig_grid=torch.zeros(out_height,out_width,2):type(self.coefficients:type());
   for i=1,out_height do
     orig_grid:select(3,1):select(1,i):fill(-1 + (i-1)/(out_height-1) * 2)
   end
   for j=1,out_width do
     orig_grid:select(3,2):select(2,j):fill(-1 + (j-1)/(out_width-1) * 2)
   end
   orig_grid=orig_grid:resize(out_height*out_width,2);
   orig_grid=orig_grid:t();
   -- local source_pts=baseGrid_t:t();

   -- print ('in tps orig_grid:size()',orig_grid:size());
   -- print ('in tps source_pts:size()',source_pts:size())
   local num_orig=orig_grid:size(2);
   local num_sp=source_pts:size(2);
   local s_p_x=torch.repeatTensor(source_pts:select(1,1):clone():view(source_pts:size(2),1),1,num_orig);
   local s_p_y=torch.repeatTensor(source_pts:select(1,2):clone():view(source_pts:size(2),1),1,num_orig);
   local o_g_x=torch.repeatTensor(orig_grid:select(1,1):clone(),num_sp,1);
   local o_g_y=torch.repeatTensor(orig_grid:select(1,2):clone(),num_sp,1);
   local distances_new=torch.pow(s_p_x-o_g_x,2)+torch.pow(s_p_y-o_g_y,2);
   distances_new= distances_new:cmul(torch.log(distances_new));
   distances_new[distances_new:ne(distances_new)]=0;
   distances_new[distances_new:eq(3/0)]=0;
   local upper_array=torch.ones(1,orig_grid:size(2)):type(self.coefficients:type());
   upper_array=torch.cat({upper_array,orig_grid},1);
   local right_mat=torch.cat({upper_array,distances_new},1);

   local transformed_points=torch.bmm(self.coefficients,torch.repeatTensor(right_mat,self.coefficients:size(1),1,1));


   -- print ('coeff size',self.coefficients:size())
   -- print ('right_mat size',right_mat:size());
   -- print ('orig_grid size',orig_grid:size());
   -- print ('source_pts size',source_pts:size());

   local x_t_all=torch.Tensor(torch.LongStorage{transformed_points:size(1),out_height,out_width}):type(transformed_points:type());
   local y_t_all=torch.Tensor(torch.LongStorage{transformed_points:size(1),out_height,out_width}):type(transformed_points:type());

   for idx=1,transformed_points:size(1) do
    x_t_all[idx]=transformed_points[idx][1]:resize(out_height,out_width);
    y_t_all[idx]=transformed_points[idx][2]:resize(out_height,out_width);
   end

   local output=torch.cat(x_t_all,y_t_all,4);
   return output;

end

function TGG:updateOutputSpecific(dest_points,batch_idx)
   local dest_points = dest_points:clone();
   local source_pts = self.source_points:clone();
   
   local distances =torch.zeros(source_pts:size(2),dest_points:size(2)):type(dest_points:type());

   for ps_idx=1,source_pts:size(2) do
     for po_idx=1,dest_points:size(2) do
       distances[{ps_idx,po_idx}]=_U_func(source_pts[{1,ps_idx}],source_pts[{2,ps_idx}],dest_points[{1,po_idx}],dest_points[{2,po_idx}]);
     end
   end

   local upper_array=torch.ones(1,dest_points:size(2)):type(dest_points:type());
   upper_array=torch.cat({upper_array,dest_points},1);
   local right_mat=torch.cat({upper_array,distances},1):type(dest_points:type());
   
   local transformed_points=self.coefficients[batch_idx]*right_mat;
   return transformed_points;

end

function TGG:updateGradInput(_destOffsets,_gradGrid)
   local destOffsets, grad_grid
   local A = self.A:clone();

   if _destOffsets:nDimension()==1 then
      destOffsets = addOuterDim(_destOffsets)
      grad_grid = addOuterDim(_gradGrid)
   else
      destOffsets = _destOffsets
      grad_grid = _gradGrid
   end

   local y = grad_grid:select(4,2):clone();
   local x = grad_grid:select(4,1):clone();
   y = y:view(y:size(1),y:size(2)*y:size(3));
   x = x:view(x:size(1),x:size(2)*x:size(3));
   
   local grad_grid_rs=torch.cat(x,y,3);
   
   local A_rep=A:resize(1,A:size(1),A:size(2)):repeatTensor(grad_grid_rs:size(1),1,1);

   -- self.gradInput:resizeAs(transformMatrix):zero()
   -- self.gradInput:baddbmm
   self.gradInput:resize(A_rep:size(1),A_rep:size(2),grad_grid_rs:size(3)):zero();

   self.gradInput:baddbmm(A_rep,grad_grid_rs);
   
   self.gradInput=self.gradInput:transpose(2,3);
   self.gradInput=self.gradInput:resize(self.gradInput:size(1),2*self.gradInput:size(3));
   
   if _destOffsets:nDimension()==1 then
      self.gradInput = self.gradInput:select(1,1)
   end

   -- print (self.gradInput:type());
   -- print (self.gradInput:size());
   return self.gradInput

end
