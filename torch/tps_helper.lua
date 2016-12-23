do  
    local TPS_Helper = torch.class('TPS_Helper')

    function TPS_Helper:__init()
        
    end

    function TPS_Helper:getLoss(pred_output,gt_output,ind_std)
		local loss=torch.pow(pred_output-gt_output,2);
		-- print (ind_mean);
		if ind_mean then
			loss=torch.mean(loss:view(loss:size(1),loss:size(2)*loss:size(3)*loss:size(4)),2);
		else
			loss=torch.mean(loss);
		end
		return loss;
    end

	function TPS_Helper:getTransformedLandMarkPoints(labels_all,out_grids,tableFlag)
		local t_pts_all;
		if tableFlag then
			t_pts_all={};
			for idx=1,#labels_all do
				t_pts_all[idx]=labels_all[idx]:clone();
			end
		else
			t_pts_all=labels_all:clone();
		end

		for i=1,out_grids:size(1) do
			local grid_curr=out_grids[i];
			local labels=labels_all[i];
			local end_label;
			
			if tableFlag then
				end_label=labels:size(2);
			else
				end_label=labels:size(1);
			end
			-- print ('labels',labels)

			for label_idx=1,end_label do

				local label_curr
				if tableFlag then
					label_curr=labels[{{1,2},label_idx}];
					-- print ('label_curr',label_idx,label_curr);
				else
					label_curr=labels[{label_idx,{1,2}}];
				end
				label_curr=label_curr:view(1,1,2);
				label_curr=torch.repeatTensor(label_curr,grid_curr:size(1),grid_curr:size(2),1);
				
				local dist=torch.sum(torch.pow(grid_curr-label_curr,2),3);
				dist=dist:view(dist:size(1),dist:size(2));
				local idx=torch.find(dist:eq(torch.min(dist)),1)[1]
				local row = math.ceil(idx/dist:size(2));
				local col = idx%dist:size(2);
				if col==0 then
					col=dist:size(2);
				end
				
				if tableFlag then
					t_pts_all[i][1][label_idx]=row;
					t_pts_all[i][2][label_idx]=col;
					-- print (row,col,t_pts_all[i]);
				else
					t_pts_all[i][label_idx][1]=row;
					t_pts_all[i][label_idx][2]=col;
				end
				
			end
		end
		return t_pts_all;
	end

	function TPS_Helper:switchMeans(training_data,imagenet_mean,mean,std)
		assert (#imagenet_mean==3);
		-- for i=1,3 do
  --           img_horse[i]:csub(params.imagenet_mean[i])
  --       end
  		for i=1,3 do
  			training_data[{{},i,{},{}}]= training_data[{{},i,{},{}}]+imagenet_mean[i];
  		end

  		local mean=mean:view(1,mean:size(1),mean:size(2),mean:size(3));
  		local std=std:view(1,std:size(1),std:size(2),std:size(3));
  		mean=torch.repeatTensor(mean,training_data:size(1),1,1,1):type(training_data:type());
  		std=torch.repeatTensor(std,training_data:size(1),1,1,1):type(training_data:type());
  		training_data=torch.cdiv((training_data-mean),std);
  		return training_data;
	end

	function TPS_Helper:switchMeans_withMeanStd(training_data,imagenet_mean,mean,std)
		assert (#imagenet_mean==3);
		-- for i=1,3 do
  --           img_horse[i]:csub(params.imagenet_mean[i])
  --       end
  		for i=1,3 do
  			training_data[{{},i,{},{}}]= training_data[{{},i,{},{}}]+imagenet_mean[i];
  		end

  		-- local mean=mean:view(1,mean:size(1),mean:size(2),mean:size(3));
  		-- local std=std:view(1,std:size(1),std:size(2),std:size(3));
  		-- mean=torch.repeatTensor(mean,training_data:size(1),1,1,1):type(training_data:type());
  		-- std=torch.repeatTensor(std,training_data:size(1),1,1,1):type(training_data:type());
  		training_data=torch.cdiv((training_data-mean),std);
  		return training_data;
	end
	
	function TPS_Helper:unMean(training_data,mean,std)
  		local mean=mean:view(1,mean:size(1),mean:size(2),mean:size(3));
  		local std=std:view(1,std:size(1),std:size(2),std:size(3));
  		mean=torch.repeatTensor(mean,training_data:size(1),1,1,1):type(training_data:type());
  		std=torch.repeatTensor(std,training_data:size(1),1,1,1):type(training_data:type());
  		training_data=torch.cmul(training_data,std)+mean;
  		return training_data;
	end	

	
	function TPS_Helper:switchMeansDebug(training_data,imagenet_mean,mean,std)
		assert (#imagenet_mean==3);
		-- for i=1,3 do
  --           img_horse[i]:csub(params.imagenet_mean[i])
  --       end
  		for i=1,3 do
  			training_data[{{},i,{},{}}]= training_data[{{},i,{},{}}]+imagenet_mean[i];
  		end

  		local mean=mean:view(1,mean:size(1),mean:size(2),mean:size(3));
  		local std=std:view(1,std:size(1),std:size(2),std:size(3));
  		mean=torch.repeatTensor(mean,training_data:size(1),1,1,1):type(training_data:type());
  		std=torch.repeatTensor(std,training_data:size(1),1,1,1):type(training_data:type());
  		training_data=torch.cdiv((training_data-mean),std);
  		return training_data;
	end	


	function TPS_Helper:getPointsOriginalImage(outputs,out_grids)
	
		-- local for_loss_all=torch.zeros(outputs:size()):type(outputs:type());
		local for_loss_all=torch.zeros(outputs:size()):type(outputs:type());

		local outputs=outputs:clone();
		-- print (outputs[4]);
		outputs=outputs:resize(outputs:size(1),outputs:size(2)/2,2):transpose(2,3);
		-- print (outputs[4]);
		
		outputs[{{},1,{}}]= (outputs[{{},1,{}}]+1)/2*out_grids:size(2);
		outputs[{{},2,{}}]= (outputs[{{},2,{}}]+1)/2*out_grids:size(3);
		-- print (outputs[4]);
		outputs=torch.round(outputs);
		

		for idx_im=1,outputs:size(1) do
			for label_idx=1,outputs:size(3) do

				local r_idx=outputs[idx_im][1][label_idx];
				-- math.round(outputs[idx_im][1][label_idx]);
				local c_idx=outputs[idx_im][2][label_idx];
				-- math.floor(outputs[idx_im][2][label_idx]);
				
				if r_idx<1 then
					r_idx=1;
				end

				if c_idx<1 then
					c_idx=1;
				end

				if r_idx>out_grids:size(2) then
					r_idx=out_grids:size(2);
				end

				if c_idx>out_grids:size(3) then
					c_idx=out_grids:size(3);
				end
			
				local grid_value_r=out_grids[idx_im][r_idx][c_idx][1];
				local grid_value_c=out_grids[idx_im][r_idx][c_idx][2];

				local for_loss_r= grid_value_r;
				local for_loss_c= grid_value_c;
				for_loss_all[idx_im][(2*label_idx)-1]=for_loss_r;
				for_loss_all[idx_im][2*label_idx]=for_loss_c;
			end
		end
		return for_loss_all

	end


	function TPS_Helper:getAffineTransform(human_label,horse_label)
	    -- local mat=torch.zeros(2,3):type(human_label:type());
	    local M=torch.zeros(human_label:size(2)*2,6):type(human_label:type());
	    local proj=torch.zeros(human_label:size(2)*2,1):type(human_label:type());
	    for idx=1,human_label:size(2) do
	        local row_x=(idx*2)-1;
	        local row_y=(idx*2);
	        proj[row_x][1]=horse_label[1][idx];
	        proj[row_y][1]=horse_label[2][idx];

	        M[row_x][1]=human_label[1][idx];
	        M[row_x][2]=human_label[2][idx];
	        M[row_x][3]=1;

	        M[row_y][4]=human_label[1][idx];
	        M[row_y][5]=human_label[2][idx];
	        M[row_y][6]=1;
	    end
	    local mat=torch.inverse(M:t()*M)*M:t()*proj;
	    mat=mat:view(2,3);
	    return mat;
	end

	function TPS_Helper:getGTParams(human_labels,horse_labels)
	    local transform_params=torch.zeros(#human_labels,2,3):type(human_labels[1]:type());
	    for idx=1,#human_labels do
	        local human_label=human_labels[idx];
	        local horse_label=horse_labels[idx];
	        
	        assert (human_label:size(2)==horse_label:size(2));
	        transform_params[idx]=self:getAffineTransform(human_label,horse_label);
	        
	    end
	    return transform_params;
	end


end    

return TPS_Helper;