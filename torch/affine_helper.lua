do  
    local Affine_Helper = torch.class('Affine_Helper')

    function Affine_Helper:__init()
        
    end


    function Affine_Helper:getAffineTransform(human_label,horse_label)
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

    function Affine_Helper:getGTParams(human_labels,horse_labels)
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

return Affine_Helper;