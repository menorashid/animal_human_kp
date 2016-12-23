
do  
    local Loss_Helper = torch.class('Loss_Helper')

    function Loss_Helper:getLossD_RCNN(pred_output_all,gt_output_all)
        local loss_all=torch.zeros(pred_output_all:size()):type(pred_output_all:type());
        for im_idx=1,pred_output_all:size(1) do
            local gt_output=gt_output_all[im_idx];
            local pred_output=pred_output_all[im_idx];
            local gt_index = gt_output[{{},3}];
            local gt_output = gt_output[{{},{1,2}}]:clone();
            gt_output=gt_output:view(gt_output:nElement());            
            local x = pred_output-gt_output;
            local loss = torch.zeros(x:size(1),1);
            for i=1,loss:size(1) do
                local loss_curr=x[i];
                if gt_index[math.floor((i+1)/2)]<0 then
                    loss[i]=0;
                else
                    if torch.abs(loss_curr)<1 then
                        loss[i]=loss_curr;
                    elseif loss_curr<0 then
                        loss[i]=-1;
                    else
                        loss[i]=1;
                    end
                end
            end
            loss_all[im_idx]=loss;
        end
        return loss_all;
    end

    function Loss_Helper:getLoss_RCNN(pred_output_all,gt_output_all)
        local loss_all=torch.zeros(pred_output_all:size(1)):type(pred_output_all:type());
        for im_idx=1,pred_output_all:size(1) do
            local gt_output=gt_output_all[im_idx];
            local pred_output=pred_output_all[im_idx];
            local gt_index=gt_output[{{},3}];
            local gt_output=gt_output[{{},{1,2}}]:clone();
            gt_output=gt_output:view(gt_output:nElement());
            local x = torch.abs(pred_output-gt_output);
            local loss = torch.zeros(x:size(1));
            local loss_total=0;
            for i=1,loss:size(1) do
                local loss_curr=x[i];
                if gt_index[math.floor((i+1)/2)]<0 then
                    loss[i]=0;
                else
                    if loss_curr<1 then
                        loss[i]=loss_curr*loss_curr*0.5;
                    else
                        loss[i]=loss_curr-0.5;
                    end
                    loss_total=loss_total+loss[i];
                end
            end
            loss = loss_total/torch.sum(gt_index:gt(0));
            loss_all[im_idx]=loss;
        end
        return torch.mean(loss_all),loss_all;
    end


    function Loss_Helper:getLossD_Euclidean(pred_output_all,gt_output_all)
        print ('TO DO');
    end

    function Loss_Helper:getLoss_EuclideanTPS(pred_output_all,gt_output_all)
        local loss_all=torch.zeros(#pred_output_all):type(pred_output_all[1]:type());
        for im_idx=1,#pred_output_all do
            local gt_output=gt_output_all[im_idx];
            local pred_output=pred_output_all[im_idx];
            local loss=torch.pow(torch.sum(torch.pow(pred_output-gt_output,2),1),0.5);
            loss_all[im_idx]=torch.mean(loss);
        end
        return torch.mean(loss_all),loss_all;
    end

    function Loss_Helper:getLoss_Euclidean(pred_output_all,gt_output_all)
        local loss_all=torch.zeros(pred_output_all:size(1)):type(pred_output_all:type());

        for im_idx=1,pred_output_all:size(1) do

            local gt_output=gt_output_all[im_idx];
            local pred_output=pred_output_all[im_idx];

            local gt_index=gt_output[{{},3}];
            local idx_keep=torch.find(gt_index:gt(0), 1)
            local gt_output=gt_output[{{},{1,2}}]:clone();

            pred_output=pred_output:view(pred_output:size(1)/2,2);
            local loss=torch.pow(torch.sum(torch.pow(pred_output-gt_output,2),2),0.5);
            local loss_keep=torch.zeros(#idx_keep):type(pred_output:type());
            for idx_idx_keep=1,#idx_keep do
                loss_keep[idx_idx_keep]=loss[idx_keep[idx_idx_keep]];
            end
            loss_all[im_idx]=torch.mean(loss_keep);
        end
        return torch.mean(loss_all),loss_all;
    end

    function Loss_Helper:getLossD_L2(pred_output,gt_output)
        local lossD=pred_output-gt_output
        lossD=torch.mul(lossD,2);
        return lossD;
    end

    function Loss_Helper:getLoss_L2(pred_output,gt_output)
        local loss=torch.pow(pred_output-gt_output,2);
        loss=torch.mean(loss);
        return loss;
    end
    
end

return Loss_Helper