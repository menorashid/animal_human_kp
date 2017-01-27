
do  
    local data = torch.class('data_animal')

    function data:__init(args)
        self.file_path=args.file_path;
        self.batch_size=args.batch_size;
        self.mean_file=args.mean_file;
        self.std_file=args.std_file;
        self.limit=args.limit;
        self.augmentation=args.augmentation;
        self.rotFix=args.rotFix;
        self.bgr=args.bgr;
        
        print ('self.augmentation',self.augmentation);

        self.start_idx_horse=1;
        self.params={input_size={40,40},
                    labels_shape={5,3},
                    angles={-10,-5,0,5,10}};

        if args.input_size then
            self.params.input_size=args.input_size
        end

        if args.imagenet_mean then
           self.params.imagenet_mean={122,117,104};
        end 
        
        self.mean_im=image.load(self.mean_file)*255;
        self.std_im=image.load(self.std_file)*255;



        self.training_set={};
        
        self.lines_horse=self:readDataFile(self.file_path);
        
        if self.augmentation then
            self.lines_horse =self:shuffleLines(self.lines_horse);
        end



        if self.limit~=nil then
            local lines_horse=self.lines_horse;
            self.lines_horse={};

            for i=1,self.limit do
                self.lines_horse[#self.lines_horse+1]=lines_horse[i];
            end
        end
        print (#self.lines_horse);
    end


    function data:shuffleLines(lines)
        local x=lines;
        local len=#lines;

        local shuffle = torch.randperm(len)
        
        local lines2={};
        for idx=1,len do
            lines2[idx]=x[shuffle[idx]];
        end
        return lines2;
    end

    function data:getTrainingData()
        local start_idx_horse_before = self.start_idx_horse

        self.training_set.data=torch.zeros(self.batch_size,3,self.params.input_size[1]
            ,self.params.input_size[2]);
        self.training_set.label=torch.zeros(self.batch_size,self.params.labels_shape[1],self.params.labels_shape[2]);
        self.training_set.input={};
        
        self.start_idx_horse=self:addTrainingData(self.training_set,self.batch_size,
            self.lines_horse,self.start_idx_horse,self.params)    
        
        

        if self.start_idx_horse<start_idx_horse_before and self.augmentation then
            print ('shuffling data'..self.start_idx_horse..' '..start_idx_horse_before )
            self.lines_horse=self:shuffleLines(self.lines_horse);
        end

    end

    
    function data:readDataFile(file_path)
        local file_lines = {};
        for line in io.lines(file_path) do 
            local start_idx, end_idx = string.find(line, ' ');
            local img_path=string.sub(line,1,start_idx-1);
            local img_label=string.sub(line,end_idx+1,#line);
            file_lines[#file_lines+1]={img_path,img_label};
        end 
        return file_lines

    end

    function data:hFlipImAndLabel(im,label)
        
        if im then
            image.hflip(im,im);
        end

        label[{{},2}]=-1*label[{{},2}]
        local temp=label[{1,{}}]:clone();
        label[{1,{}}]=label[{2,{}}]:clone()
        label[{2,{}}]=temp;

        temp=label[{4,{}}]:clone();
        label[{4,{}}]=label[{5,{}}]:clone()
        label[{5,{}}]=temp;
        
        return im,label;

    end

    function data:rotateImAndLabel(img_horse,label_horse,angles)
        
        local isValid = false;
        local img_horse_org=img_horse:clone();
        local label_horse_org=label_horse:clone();
        local iter=0;
        
        while not isValid do
            isValid = true;
            label_horse= label_horse_org:clone();

            local rand=math.random(#angles);
            local angle=math.rad(angles[rand]);
            img_horse=image.rotate(img_horse_org,angle,"bilinear");

            if self.rotFix then
                local rot = torch.ones(img_horse_org:size());
                rot=image.rotate(rot,angle,"simple");
                img_horse[rot:eq(0)]=img_horse_org[rot:eq(0)];
            end

            local rotation_matrix=torch.zeros(2,2);
            rotation_matrix[1][1]=math.cos(angle);
            rotation_matrix[1][2]=math.sin(angle);
            rotation_matrix[2][1]=-1*math.sin(angle);
            rotation_matrix[2][2]=math.cos(angle);
            
            for i=1,label_horse:size(1) do
                if label_horse[i][3]>0 then
                    local ans = rotation_matrix*torch.Tensor({label_horse[i][2],label_horse[i][1]}):view(2,1);
                    label_horse[i][1]=ans[2][1];
                    label_horse[i][2]=ans[1][1];
                    if torch.all(label_horse[i]:ge(-1)) and torch.all(label_horse[i]:le(1)) then
                        isValid=true;
                    else
                        isValid=false;
                    end
                end

                if not isValid then
                    break;
                end

            end
            
            iter=iter+1;
            if iter==100 then
                print ('BREAKING rotation');
                label_horse=label_horse_org;
                img_horse=img_horse_org;
                break;
            end
        end

        return img_horse,label_horse
    end

    

    function data:processImAndLabel(img_horse,label_horse,params)
        
        img_horse:mul(255);
        
        local org_size_horse=img_horse:size();
        local label_horse_org=label_horse:clone();
        
        if img_horse:size(2)~=params.input_size[1] then 
            img_horse = image.scale(img_horse,params.input_size[1],params.input_size[2]);
        end
        label_horse[{{},1}]=label_horse[{{},1}]/org_size_horse[3]*params.input_size[1];
        label_horse[{{},1}]=(label_horse[{{},1}]/params.input_size[1]*2)-1;
        label_horse[{{},2}]=label_horse[{{},2}]/org_size_horse[2]*params.input_size[2];
        label_horse[{{},2}]=(label_horse[{{},2}]/params.input_size[2]*2)-1;
        
        local temp = label_horse[{{},1}]:clone();
        label_horse[{{},1}]=label_horse[{{},2}]
        label_horse[{{},2}]=temp

        if (torch.max(label_horse)>=params.input_size[1]) then
            print ('PROBLEM horse');
            print (label_horse_org);
            print (label_horse);
            print (org_size_horse);
        end

        if self.params.imagenet_mean then
            for i=1,img_horse:size()[1] do
                img_horse[i]:csub(params.imagenet_mean[i])
            end
        else
            img_horse=torch.cdiv((img_horse-self.mean_im),self.std_im);
        end
        
        -- flip or rotate
        if self.augmentation then
            local rand=math.random(2);
            if rand==1 then
                img_horse,label_horse = self:hFlipImAndLabel(img_horse,label_horse);
            end
            img_horse,label_horse=self:rotateImAndLabel(img_horse,label_horse,params.angles);
        end

        return img_horse,label_horse
    end

    function data:addTrainingData(training_set,batch_size,lines_horse,start_idx_horse,params)
        local list_idx=start_idx_horse;
        local list_size=#lines_horse;
        local curr_idx=1;
        while curr_idx<= batch_size do
            local img_path_horse=lines_horse[list_idx][1];
            local label_path_horse=lines_horse[list_idx][2];
            
            local status_img_horse,img_horse=pcall(image.load,img_path_horse);
            
            if status_img_horse then
                local label_horse=npy4th.loadnpy(label_path_horse):double();

                if img_horse:size()[1]==1 then
                    img_horse= torch.cat(img_horse,img_horse,1):cat(img_horse,1)
                end
                img_horse,label_horse=self:processImAndLabel(img_horse,label_horse,params)
                if self.bgr then
                    -- print 'bgring';
                    local img_horse_temp=img_horse:clone();
                    img_horse[{1,{},{}}]=img_horse_temp[{3,{},{}}];
                    img_horse[{3,{},{}}]=img_horse_temp[{1,{},{}}];
                end
                training_set.data[curr_idx]=img_horse:int();
                training_set.label[curr_idx]=label_horse;
                training_set.input[curr_idx]=img_path_horse;
            else
                print ('PROBLEM READING INPUT');
                curr_idx=curr_idx-1;
            end
            list_idx=(list_idx%list_size)+1;
            curr_idx=curr_idx+1;
        end
        return list_idx;
    end

    
end

return data