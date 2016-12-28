
do  
    local Forward_Helper = torch.class('Forward_Helper')

    function Forward_Helper:__init()
        
    end

	function Forward_Helper:forward(td,net,inputs,batch_targets,saveImage,euclideanLoss,mean_im,std_im)
	    local batch_inputs_view;
	    if saveImage then
	        batch_inputs_view=inputs:double():clone();
	    end  

	    local midoutputs=net:get(1):forward(inputs);

	    local midoutputs_view;
	    if saveImage then
	        midoutputs_view=midoutputs:double():clone();
	    end

	    if not mean_im then
	    	mean_im=td.mean_im;
	    end

	    if not std_im then 
	    	std_im=td.std_im;
	    end

        midoutputs=tps_helper:switchMeans(midoutputs,td.params.imagenet_mean,mean_im,std_im)

	    local outputs=net:get(2):forward(midoutputs);
	    
	    local tps_layer= net:get(1):get(1):get(2);
	    tps_layer=tps_layer:get(#tps_layer);
	    

	    local t_pts=tps_helper:getPointsOriginalImage(outputs,tps_layer.output)

	    if saveImage then
	        local outputs_view=outputs:view(outputs:size(1),outputs:size(2)/2,2):clone();
	        local t_pts_view=t_pts:view(t_pts:size(1),t_pts:size(2)/2,2):clone();
	        local colors={{0,255,0}};
	        local pointSize=10;  
	        t_pts_view=t_pts_view:transpose(2,3)
	        for im_num=1,t_pts:size(1) do 
	            local out_file_gt=saveImage..im_num..'_gt_pts.npy';
	            local out_file_pred=saveImage..im_num..'_pred_pts.npy';

	            local pred_output=t_pts[im_num]:clone():double();
	            local gt_output=batch_targets[im_num]:clone():double();
	            pred_output=pred_output:view(pred_output:size(1)/2,2);
	            npy4th.savenpy(out_file_gt,gt_output);
	            npy4th.savenpy(out_file_pred,pred_output);

	        end

	        local binary=batch_targets[{{},{},3}]:clone();

	        visualize:saveBatchImagesWithKeypointsSensitive(batch_inputs_view,t_pts_view,{saveImage,'_org.jpg'},td.params.imagenet_mean,{-1,1},colors,pointSize,binary);

	        visualize:saveBatchImagesWithKeypointsSensitive(batch_inputs_view,batch_targets[{{},{},{1,2}}]:transpose(2,3),{saveImage,'_gt.jpg'},nil,{-1,1},colors,pointSize,binary);

	        visualize:saveBatchImagesWithKeypointsSensitive(midoutputs_view,outputs_view:transpose(2,3),{saveImage,'_warp.jpg'},td.params.imagenet_mean,{-1,1},colors,pointSize,binary);

	        visualize:saveBatchImagesWithKeypointsSensitive(batch_inputs_view,t_pts_view,{saveImage,'_org_nokp.jpg'},nil,{-1,1},colors,-1,binary);

	        visualize:saveBatchImagesWithKeypointsSensitive(midoutputs_view,outputs_view:transpose(2,3),{saveImage,'_warp_nokp.jpg'},nil,{-1,1},colors,-1,binary);
	    end

	    local loss;
	    local dloss;

	    if euclideanLoss then
	        loss, loss_all = loss_helper:getLoss_Euclidean(t_pts,batch_targets);
	        dloss=loss_all;
	    else
	        dloss = loss_helper:getLossD_RCNN(t_pts,batch_targets);
	        loss = loss_helper:getLoss_RCNN(t_pts,batch_targets);
	    end

	    return loss,dloss,midoutputs,inputs,tps_layer;    
	end

	function Forward_Helper:forward_noWarp(td,net,batch_inputs,batch_targets,saveImage)
	    local outputs=net:forward(batch_inputs);
	    loss,loss_all = loss_helper:getLoss_Euclidean(outputs,batch_targets);
	    
	    if saveImage then
	        local outputs_view=outputs:view(outputs:size(1),outputs:size(2)/2,2):clone();
	        local batch_inputs_view=batch_inputs:clone():double();
	        batch_inputs_view=tps_helper:unMean(batch_inputs_view,td.mean_im,td.std_im);
	        
	        for im_num=1,outputs_view:size(1) do 
	            local out_file_gt=saveImage..im_num..'_gt_pts.npy';
	            local out_file_pred=saveImage..im_num..'_pred_pts.npy';

	            local pred_output=outputs_view[im_num]:clone():double();
	            local gt_output=batch_targets[im_num]:clone():double();
	            npy4th.savenpy(out_file_gt,gt_output);
	            npy4th.savenpy(out_file_pred,pred_output);
	        end
	    
	        local binary=batch_targets[{{},{},3}]:clone();
	    
	        visualize:saveBatchImagesWithKeypointsSensitive(batch_inputs_view,outputs_view:transpose(2,3),{saveImage,'_org.jpg'},nil,{-1,1},colors,pointSize,binary);
	    
	        visualize:saveBatchImagesWithKeypointsSensitive(batch_inputs_view,batch_targets[{{},{},{1,2}}]:transpose(2,3),{saveImage,'_gt.jpg'},nil,{-1,1},colors,pointSize,binary);

	        visualize:saveBatchImagesWithKeypointsSensitive(batch_inputs_view,outputs_view:transpose(2,3),{saveImage,'_org_nokp.jpg'},nil,{-1,1},colors,-1,binary);
	    end

	    return loss,loss_all;
	end

end    

return Forward_Helper;