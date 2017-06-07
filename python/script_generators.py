import util;
import numpy as np;
import os;
import shutil;
import itertools;
import visualize_results as viz;
import visualize;

def rerun_5040_bl_ft():
    out_dir_meta='/home/SSD3/maheen-data/horse_project/face_baselines_small_data_rerun_50_dec';
    out_file_sh='../scripts/rerun_5040_bl_ft.sh';

    num_neighbors=range(5,10,5);
    num_data=range(500,3500,500);
    num_data=num_data+[3531]
    post_tags=['_horse.txt','_face.txt','_face_noIm.txt'];
    # cmd:option('-model','/disk2/horse_cvpr/ft_kp_imagenet/model_17_mod.dat');
    dir_neighbors='/home/SSD3/maheen-data/horse_project/neighbor_data';
    out_dir_breakdowns=os.path.join(dir_neighbors,'small_datasets');
    file_pre='matches';
    minloss_post='_minloss.txt';
    num_files=2;
    
    out_file_pres=[];
    for num_neighbor in num_neighbors:
        for num_data_curr in num_data:
            file_curr=file_pre+'_'+str(num_neighbor)+'_'+str(num_data_curr);
            out_file_pre=os.path.join(out_dir_breakdowns,file_curr);
            out_file_pres.append(out_file_pre);

    train_data_paths=[file_curr+post_tags[0][:post_tags[0].rindex('.')]+minloss_post for file_curr in out_file_pres];
    out_dirs=[os.path.join(out_dir_meta,os.path.split(train_data_path)[1][:-4]) for train_data_path in train_data_paths];
    
    commands=[];
    for idx_out_dir_curr,out_dir_curr in enumerate(out_dirs):
        out_dir_out=os.path.join(out_dir_curr,'resume_50');
        command_curr=[];
        command_curr.extend(['th','train_kp_net.th']);
        command_curr.extend(['-model',os.path.join(out_dir_curr,'intermediate','model_all_5040.dat')]);
        command_curr.extend(['-outDir',out_dir_out]);
        command_curr.extend(['-mean_im_path','../data/aflw_cvpr_224_mean.png']);
        command_curr.extend(['-std_im_path','../data/aflw_cvpr_224_std.png']);
        command_curr.extend(['-iterations',str(56*60)]);
        command_curr.extend(['decreaseAfter',str(56*10)]);
        command_curr.extend(['learningRate',str(1e-3)]);
        command_curr.extend(['-data_path',train_data_paths[idx_out_dir_curr]]);
        command_curr=' '.join(command_curr);
        commands.append(command_curr);
    print out_file_sh
    util.writeFile(out_file_sh,commands);


def copy_files_ext_disk():
    dir_meta_ext='/home/SSD3/maheen-data/horse_project/ext_disk/face_baselines_small_data';
    dir_meta_home='/home/SSD3/maheen-data/horse_project/face_baselines_small_data_rerun_50_dec';
    dir_curr_pre='matches_5_';
    dir_curr_post='_horse_minloss';
    num_data=range(500,3500,500);
    num_data=num_data+[3531]
    dirs=[dir_curr_pre+str(num_data_curr)+dir_curr_post for num_data_curr in num_data];

    post_files=[os.path.join('final','model_all_final.dat'),os.path.join('intermediate','model_all_5040.dat')];
    all_files_post=[os.path.join(dir_curr,post_file_curr) for dir_curr in dirs for post_file_curr in post_files];
    commands=[];
    for all_file_post in all_files_post:
        in_file=os.path.join(dir_meta_ext,all_file_post);
        out_file=os.path.join(dir_meta_home,all_file_post);
        out_dir_curr=os.path.split(out_file)[0];
        
        if not os.path.exists(out_dir_curr):
            os.makedirs(out_dir_curr);

        # if not os.path.exists(out_file):
        #     shutil.copy(in_file,out_file);

        # print in_file,os.path.exists(in_file);
        # print out_file,os.path.exists(out_file);
        commands.append('cp '+in_file+' '+out_file);

    util.writeFile('../scripts/copy_ext_disk_rerun.sh',commands);


def rerun_5040_bl_alexnet():
    # script_make_small_data_paths();
    out_file_script='../scripts/alexnet_rerun_50';
    num_scripts=1;
    torch_file='train_kp_alexnet.th';
    out_dir_meta='/home/SSD3/maheen-data/horse_project/cvpr_rebuttal/imagenet_last_scratch_small_data';
    util.mkdir(out_dir_meta);
    train_data_dir='../data_227';
    train_file_pre='small_train_minloss_'
    range_data=range(500,3500,500);
    range_data.append(3531);

    # train_file_pre=
    # os.path.join(train_data_dir,train_file_pre);

    commands=[];
    for num_data in range_data:
        dir_curr=os.path.join(out_dir_meta,train_file_pre+str(num_data));
        train_file=os.path.join(train_data_dir,train_file_pre+str(num_data)+'.txt');

        dir_out=os.path.join(dir_curr,'resume_50');
        in_model_file=os.path.join(dir_curr,'intermediate','model_all_5040.dat');

        params=['th',torch_file];
        params.extend(['-outDir',dir_out]);
        params.extend(['-data_path',train_file]);

        # model path
        params.extend(['-model',in_model_file]);
        
        # learning rate
        # cmd:option('learningRate', 1e-3)
        params.extend(['learningRate',str(1e-4)]);
        # iterations
        # cmd:option('-iterations',150*epoch_size,'num of iterations to run');
        params.extend(['-iterations',str(60*56)]);
        
        # decreaseAfter
        params.extend(['decreaseAfter',str(10*56)]);
        


        command_curr=' '.join(params);
        print command_curr;
        commands.append(command_curr);

    # split_num=len(commands)/num_scripts;
    commands=np.array(commands);
    commands_split=np.array_split(commands,num_scripts);
    for idx_commands,commands in enumerate(commands_split):
        out_file_script_curr=out_file_script+'_'+str(idx_commands)+'.sh';
        print idx_commands
        print out_file_script_curr
        print commands;
        util.writeFile(out_file_script_curr,commands);



def copySmallDatatoVision3():
    out_dir='../data/small_datasets';
    util.mkdir(out_dir);
    
    small_data_dir='/home/SSD3/maheen-data/horse_project/neighbor_data/small_datasets'
    to_replace=['/home/SSD3/maheen-data/horse_project/data_check','../data'];

    file_pre=['matches'];
    num_neighbors=[str(num_curr) for num_curr in [10,15,20]];
    num_data=[3531]
    # range(500,3500,500)+[3531];
    num_data=[str(num_data_curr) for num_data_curr in num_data];
    file_post=['horse','face','face_noIm'];
    file_type=['minloss.txt','.txt'];
    # file_post=['_'+file_post_curr+file_type for file_post_curr in file_post for file_type_curr in file_type]; 
    files=['_'.join(file_curr) for file_curr in itertools.product(file_pre,num_neighbors,num_data,file_post,file_type)];
    files=[file_curr.replace('_.txt','.txt') for file_curr in files];
    for file_curr in files:
        in_file=os.path.join(small_data_dir,file_curr);
        lines=util.readLinesFromFile(in_file);
        lines=[line_curr.replace(to_replace[0],to_replace[1]) for line_curr in lines];
        out_file=os.path.join(out_dir,file_curr);
        util.writeFile(out_file,lines);
        print out_file;
        # print lines[50];
        # break;


    # print files[0];

def script_tpsUs(vision1=False):
    # num_data=range(500,3500,500);
    num_data=[3000,500,2500,1000,2000,1500,3531]
    file_data_pre='../data/small_datasets/matches_5_';
    if vision1:
        out_dir_pre_pre='/home/SSD3/maheen-data/horse_project'
    else:
        out_dir_pre_pre='/disk2/horse_cvpr'
    out_dir_pre=os.path.join(out_dir_pre_pre,'tps_nets_no_bn_fix/horse_')
    torch_file='train_warping_net.th';
    out_file_script='../scripts/train_warping_small_data_no_bn_fix';
    num_scripts=2;
    commands=[];
    for num_data_curr in num_data:
        command_curr=['th',torch_file];
        command_curr.extend(['-outDir',out_dir_pre+str(num_data_curr)]);
        command_curr.extend(['-horse_data_path',file_data_pre+str(num_data_curr)+'_horse.txt']);
        command_curr.extend(['-human_data_path',file_data_pre+str(num_data_curr)+'_face_noIm.txt']);
        command_curr.extend(['-model','../models/tps_localization_net_untrained.dat']);
        command_curr=' '.join(command_curr);
        commands.append(command_curr);

    commands=np.array(commands);
    commands_split=np.array_split(commands,num_scripts);
    for idx_commands,commands in enumerate(commands_split):
        out_file_script_curr=out_file_script+'_'+str(idx_commands)+'.sh';
        print idx_commands
        print out_file_script_curr
        print commands;
        util.writeFile(out_file_script_curr,commands);


def script_full_us_tps(vision1=False):

    # num_data=range(500,3500,500);
    # num_data=[3000,500,2500,1000,2000,1500,3531]
    num_data=[3531];
    file_data_pre='../data/small_datasets/matches_5_';
    if vision1:
        out_dir_pre_pre='/home/SSD3/maheen-data/horse_project'
    else:
        out_dir_pre_pre='/disk2/horse_cvpr'
    out_dir_pre=os.path.join(out_dir_pre_pre,'full_nets_us_no_bn_fix/horse_')
    out_dir_tps_meta=os.path.join(out_dir_pre_pre,'tps_nets_no_bn_fix/horse_')
    torch_file='train_full_model.th';
    out_file_script='../scripts/train_full_model_us_small_data';
    num_scripts=1;
    commands=[];
    

    for num_data_curr in num_data:
        command_curr=['th',torch_file];
        command_curr.extend(['learningRate',str(1e-2)]);
        command_curr.extend(['multiplierMid',str(0.1)]);
        command_curr.extend(['multiplierBottom',str(0.01)]);
        command_curr.extend(['-horse_data_path',file_data_pre+str(num_data_curr)+'_horse.txt']);
        command_curr.extend(['-human_data_path',file_data_pre+str(num_data_curr)+'_face_noIm.txt']);
        command_curr.extend(['-numDecrease',str(2)]);
        command_curr.extend(['-tps_model_path',os.path.join(out_dir_tps_meta+str(num_data_curr),'final','model_all_final.dat')]);
        command_curr.extend(['-outDir',out_dir_pre+str(num_data_curr)]);
        command_curr.extend(['-bgr']);
        command_curr.extend(['dual']);
        command_curr=' '.join(command_curr);
        commands.append(command_curr);

    commands=np.array(commands);
    commands_split=np.array_split(commands,num_scripts);
    for idx_commands,commands in enumerate(commands_split):
        out_file_script_curr=out_file_script+'_'+str(idx_commands)+'.sh';
        print idx_commands
        print out_file_script_curr
        print commands;
        util.writeFile(out_file_script_curr,commands);

def script_bl_tps(vision1=False):

    # num_data=range(500,3500,500)+[3531];
    num_data=[3531]
    # num_data=[2000]
    file_data_pre='../data/small_datasets/matches_5_';
    if vision1:
        out_dir_pre_pre='/home/SSD3/maheen-data/horse_project'
    else:
        out_dir_pre_pre='/disk2/horse_cvpr'
    out_dir_pre=os.path.join(out_dir_pre_pre,'bl_tps_small_data_no_bn_fix_0.001_0.1_0.01/horse_')
    torch_file='train_full_model.th';
    out_file_script='../scripts/bl_tps_small_data_no_bn_fix_0.001_0.1_0.01';
    tps_model_path='../models/tps_localization_net_untrained_withTPS.dat'
    num_scripts=2;
    commands=[];
    epoch_size=56;

    for num_data_curr in num_data:
        command_curr=['th',torch_file];
        # if num_data_curr<=2000:
        command_curr.extend(['learningRate',str(1e-3)]);
        # else:
        #     command_curr.extend(['learningRate',str(1e-2)]);
        command_curr.extend(['multiplierMid',str(0.1)]);
        command_curr.extend(['multiplierBottom',str(0.01)]);
        
        command_curr.extend(['-iterations',str(200*epoch_size)]);
        command_curr.extend(['-decreaseAfter',str(100*epoch_size)]);

        command_curr.extend(['-horse_data_path',file_data_pre+str(num_data_curr)+'_horse_minloss.txt']);
        command_curr.extend(['-human_data_path',file_data_pre+str(num_data_curr)+'_face_noIm_minloss.txt']);
        command_curr.extend(['-numDecrease',str(1)]);
        command_curr.extend(['-tps_model_path',tps_model_path]);
        command_curr.extend(['-outDir',out_dir_pre+str(num_data_curr)]);
        command_curr.extend(['-bgr']);
        command_curr=' '.join(command_curr);
        commands.append(command_curr);

    commands=np.array(commands);
    commands_split=np.array_split(commands,num_scripts);
    for idx_commands,commands in enumerate(commands_split):
        out_file_script_curr=out_file_script+'_'+str(idx_commands)+'.sh';
        print idx_commands
        print out_file_script_curr
        print commands;
        util.writeFile(out_file_script_curr,commands);

def script_bl_tps_lr_search(vision1=False):

    # num_data=range(500,3500,500)+[3531];
    # num_data=[3531,500,3000,1000,2500,1500,2000]
    num_data=[3000]
    file_data_pre='../data/small_datasets/matches_5_';
    # if vision1:
    #     out_dir_pre_pre='/home/SSD3/maheen-data/horse_project'
    # else:
    #     out_dir_pre_pre='/disk2/horse_cvpr'

    out_dir_pre_pre='../experiments';

    out_dir_pre=os.path.join(out_dir_pre_pre,'bl_tps_lr_0.01_0.1_0.01')
    torch_file='train_full_model.th';
    out_file_script='../scripts/bl_tps_last_try';
    tps_model_path='../models/tps_localization_net_untrained_withTPS.dat'
    num_scripts=2;
    commands=[];
    epoch_size=56;
    # lr=[0.01,0.001,0.0001];
    # mult_mid=[1,0.1,0.01]
    # mult_b=[1,0.1];
    # # 1,
    # combs=[(lr_curr,m_m,m_m*m_b) for lr_curr in lr for m_m in mult_mid for m_b in mult_b];
    
    # for comb in combs:
    #     print comb;
    # print len(combs);

    combs=[(0.01,0.1,0.01)]
    # combs=[(0.0001,1,0.01)]
    # ,\
    #         (0.01,0.01,0.001),\
    #         (0.001,0.1,0.01)];

    out_dirs=[];
    for num_data_curr in num_data:
        for lr,mm,mb in combs:
            out_dir_curr=os.path.join(out_dir_pre,'horse_'+str(num_data_curr))
            # os.path.join(out_dir_pre,'_'.join([str(num) for num in [lr,mm,mb]]))
            command_curr=['th',torch_file];
            command_curr.extend(['learningRate',lr]);
            command_curr.extend(['multiplierMid',mm]);
            command_curr.extend(['multiplierBottom',mb]);
            command_curr.extend(['-iterations',str(200*epoch_size)]);
            command_curr.extend(['-decreaseAfter',str(100*epoch_size)]);

            command_curr.extend(['-horse_data_path',file_data_pre+str(num_data_curr)+'_horse_minloss.txt']);
            command_curr.extend(['-human_data_path',file_data_pre+str(num_data_curr)+'_face_noIm_minloss.txt']);
            command_curr.extend(['-numDecrease',str(1)]);
            command_curr.extend(['-tps_model_path',tps_model_path]);
            command_curr.extend(['-outDir',out_dir_curr]);
            command_curr.extend(['-bgr']);
            command_curr=[str(c) for c in command_curr];
            command_curr=' '.join(command_curr);
            commands.append(command_curr);
            out_dirs.append(out_dir_curr);
    # stats=[];
    # for idx_curr,out_dir_curr in enumerate(out_dirs):
    #     stat_curr=float(util.readLinesFromFile(os.path.join(out_dir_curr,'test_images','stats.txt'))[-1])
    #     print out_dir_curr,combs[idx_curr],stat_curr
    #     stats.append(stat_curr);

    # stats=np.array(stats);
    # # print stats;
    # print np.min(stats),np.argmin(stats);
    # print combs[np.argmin(stats)];
    # # print out_dirs[np.argmin(stats)];

    commands=np.array(commands);
    commands_split=np.array_split(commands,num_scripts);
    for idx_commands,commands in enumerate(commands_split):
        out_file_script_curr=out_file_script+'_'+str(idx_commands)+'.sh';
        # print idx_commands
        print out_file_script_curr
        # print commands;
        util.writeFile(out_file_script_curr,commands);

def script_bl_tps_smooth(vision1=False):
    # num_data=range(500,3500,500)+[3531];
    num_data=[3531]
    # ,500,3000,1000,2500,1500,2000]
    file_data_pre='../data/small_datasets/matches_5_';
    if vision1:
        out_dir_pre_pre='/home/SSD3/maheen-data/horse_project'
    else:
        out_dir_pre_pre='/disk2/horse_cvpr'
    out_dir_pre=os.path.join(out_dir_pre_pre,'bl_tps_lr_0.01_0.1_0.01/horse_')
    torch_file='train_full_model.th';
    out_file_script='../scripts/bl_tps_smooth';
    tps_model_path='../models/tps_localization_net_untrained_withTPS.dat'
    num_scripts=2;
    commands=[];
    epoch_size=56;
    num_runs =4;
    for run_num in range(num_runs):
        for num_data_curr in num_data:
            command_curr=['th',torch_file];
            out_dir_in=out_dir_pre+str(num_data_curr);
            out_dir_out=os.path.join(out_dir_in,'resume_180_'+str(run_num));
            command_curr.extend(['learningRate',str(1e-3)]);
            command_curr.extend(['multiplierMid',str(0.1)]);
            command_curr.extend(['multiplierBottom',str(0.01)]);
            
            command_curr.extend(['-iterations',str(20*epoch_size)]);
            command_curr.extend(['-decreaseAfter',str(60*epoch_size)]);

            command_curr.extend(['-horse_data_path',file_data_pre+str(num_data_curr)+'_horse_minloss.txt']);
            command_curr.extend(['-human_data_path',file_data_pre+str(num_data_curr)+'_face_noIm_minloss.txt']);
            command_curr.extend(['-numDecrease',str(1)]);
            command_curr.extend(['-tps_model_path',tps_model_path]);

            command_curr.extend(['-outDir',out_dir_out]);

            command_curr.extend(['-bgr']);

            command_curr.extend(['-full_model_flag']);
            command_curr.extend(['-face_detection_model_path',os.path.join(out_dir_in,'intermediate','model_all_10080.dat')]);
            command_curr=' '.join(command_curr);
            commands.append(command_curr);

    commands=np.array(commands);
    commands_split=np.array_split(commands,num_scripts);
    for idx_commands,commands in enumerate(commands_split):
        out_file_script_curr=out_file_script+'_'+str(idx_commands)+'.sh';
        print idx_commands
        print out_file_script_curr
        print commands;
        util.writeFile(out_file_script_curr,commands);

    
def script_bl_tps_resume(vision1=False):
    # num_data=range(500,3500,500)+[3531];
    num_data=[3531]
    # ,500,3000,1000,2500,1500,2000]
    file_data_pre='../data/small_datasets/matches_5_';
    if vision1:
        out_dir_pre_pre='/home/SSD3/maheen-data/horse_project'
    else:
        out_dir_pre_pre='/disk2/horse_cvpr'
    out_dir_pre=os.path.join(out_dir_pre_pre,'bl_tps_small_data_no_bn_fix/horse_')
    torch_file='train_full_model.th';
    out_file_script='../scripts/bl_tps_small_data_no_bn_fix_resume';
    tps_model_path='../models/tps_localization_net_untrained_withTPS.dat'
    num_scripts=1;
    commands=[];
    epoch_size=56;

    for num_data_curr in num_data:
        command_curr=['th',torch_file];
        out_dir_in=out_dir_pre+str(num_data_curr);
        out_dir_out=os.path.join(out_dir_in,'resume_60_again');
        if num_data_curr<=2000:
            command_curr.extend(['learningRate',str(1e-5)]);
        else:
            command_curr.extend(['learningRate',str(1e-4)]);
        command_curr.extend(['multiplierMid',str(1)]);
        command_curr.extend(['multiplierBottom',str(0.01)]);
        
        command_curr.extend(['-iterations',str(140*epoch_size)]);
        command_curr.extend(['-decreaseAfter',str(60*epoch_size)]);

        command_curr.extend(['-horse_data_path',file_data_pre+str(num_data_curr)+'_horse_minloss.txt']);
        command_curr.extend(['-human_data_path',file_data_pre+str(num_data_curr)+'_face_noIm_minloss.txt']);
        command_curr.extend(['-numDecrease',str(1)]);
        command_curr.extend(['-tps_model_path',tps_model_path]);

        command_curr.extend(['-outDir',out_dir_out]);

        command_curr.extend(['-bgr']);

        command_curr.extend(['-full_model_flag']);
        command_curr.extend(['-face_detection_model_path',os.path.join(out_dir_in,'intermediate','model_all_3360.dat')]);
        command_curr=' '.join(command_curr);
        commands.append(command_curr);

    commands=np.array(commands);
    commands_split=np.array_split(commands,num_scripts);
    for idx_commands,commands in enumerate(commands_split):
        out_file_script_curr=out_file_script+'_'+str(idx_commands)+'.sh';
        print idx_commands
        print out_file_script_curr
        print commands;
        util.writeFile(out_file_script_curr,commands);

def script_affine_us(system):

    if system=='vision4':
        out_dir_pre_pre='/data/maheen-data/horse_cvpr';
    elif system=='vision1':
        out_dir_pre_pre='/home/SSD3/maheen-data/horse_project'
    else:
        out_dir_pre_pre='/disk2/horse_cvpr'


    num_data=[3531,500,3000,1000,2500,1500,2000]
    file_data_pre='../data/small_datasets/matches_5_';
    out_dir_pre=os.path.join(out_dir_pre_pre,'affine_nets_small_data/horse_')
    torch_file='train_warping_net.th';
    out_file_script='../scripts/train_affine_warping_small_data';
    num_scripts=2;
    commands=[];
    for num_data_curr in num_data:
        command_curr=['th',torch_file];
        command_curr.extend(['-outDir',out_dir_pre+str(num_data_curr)]);
        command_curr.extend(['-horse_data_path',file_data_pre+str(num_data_curr)+'_horse.txt']);
        command_curr.extend(['-human_data_path',file_data_pre+str(num_data_curr)+'_face_noIm.txt']);
        command_curr.extend(['-model','../models/affine_localization_net_untrained.dat']);
        command_curr.extend(['affine']);

        command_curr=' '.join(command_curr);
        commands.append(command_curr);

    commands=np.array(commands);
    commands_split=np.array_split(commands,num_scripts);
    for idx_commands,commands in enumerate(commands_split):
        out_file_script_curr=out_file_script+'_'+str(idx_commands)+'.sh';
        print idx_commands
        print out_file_script_curr
        print commands;
        util.writeFile(out_file_script_curr,commands);


def script_full_us_affine(system):

    # num_data=range(500,3500,500);
    num_data=[3531,500,3000,1000,2500,1500,2000]
    file_data_pre='../data/small_datasets/matches_5_';
    if system=='vision4':
        out_dir_pre_pre='/data/maheen-data/horse_cvpr';
    elif system=='vision1':
        out_dir_pre_pre='/home/SSD3/maheen-data/horse_project'
    else:
        out_dir_pre_pre='/disk2/horse_cvpr'

    out_dir_pre=os.path.join(out_dir_pre_pre,'full_nets_us_affine_small_data/horse_')
    out_dir_tps_meta=os.path.join(out_dir_pre_pre,'affine_nets_small_data/horse_')
    torch_file='train_full_model.th';
    out_file_script='../scripts/train_full_model_us_affine_small_data';
    num_scripts=1;
    commands=[];
    

    for num_data_curr in num_data:
        command_curr=['th',torch_file];
        command_curr.extend(['learningRate',str(1e-2)]);
        command_curr.extend(['multiplierMid',str(0.1)]);
        command_curr.extend(['multiplierBottom',str(0.01)]);
        command_curr.extend(['-horse_data_path',file_data_pre+str(num_data_curr)+'_horse.txt']);
        command_curr.extend(['-human_data_path',file_data_pre+str(num_data_curr)+'_face_noIm.txt']);
        command_curr.extend(['-numDecrease',str(2)]);
        command_curr.extend(['-tps_model_path',os.path.join(out_dir_tps_meta+str(num_data_curr),'final','model_all_final.dat')]);
        command_curr.extend(['-outDir',out_dir_pre+str(num_data_curr)]);
        command_curr.extend(['-bgr']);
        command_curr.extend(['dual']);
        command_curr.extend(['-affine']);
        command_curr=' '.join(command_curr);
        commands.append(command_curr);

    commands=np.array(commands);
    commands_split=np.array_split(commands,num_scripts);
    for idx_commands,commands in enumerate(commands_split):
        out_file_script_curr=out_file_script+'_'+str(idx_commands)+'.sh';
        print idx_commands
        print out_file_script_curr
        print commands;
        util.writeFile(out_file_script_curr,commands);

def script_bl_affine(system):

    # num_data=range(500,3500,500)+[3531];
    num_data=[3531,500,3000,1000,2500,1500,2000]
    file_data_pre='../data/small_datasets/matches_5_';
    if system=='vision4':
        out_dir_pre_pre='/data/maheen-data/horse_cvpr';
    elif system=='vision1':
        out_dir_pre_pre='/home/SSD3/maheen-data/horse_project'
    else:
        out_dir_pre_pre='/disk2/horse_cvpr'

    out_dir_pre=os.path.join(out_dir_pre_pre,'bl_affine_small_data_0.001_0.1_0.01/horse_')
    torch_file='train_full_model.th';
    out_file_script='../scripts/bl_affine_small_data_fix_lr';
    tps_model_path='../models/affine_localization_net_untrained.dat'
    num_scripts=2;
    commands=[];
    epoch_size=56;

    for num_data_curr in num_data:
        command_curr=['th',torch_file];
        # if num_data_curr<=2000:
        #     command_curr.extend(['learningRate',str(1e-5)]);
        # else:
        #     command_curr.extend(['learningRate',str(1e-4)]);
        # command_curr.extend(['multiplierMid',str(1)]);
        # command_curr.extend(['multiplierBottom',str(0.01)]);

        command_curr.extend(['learningRate',str(1e-3)]);
        # else:
        #     command_curr.extend(['learningRate',str(1e-2)]);
        command_curr.extend(['multiplierMid',str(0.1)]);
        command_curr.extend(['multiplierBottom',str(0.01)]);
        
        command_curr.extend(['-iterations',str(200*epoch_size)]);
        command_curr.extend(['-decreaseAfter',str(100*epoch_size)]);

        command_curr.extend(['-horse_data_path',file_data_pre+str(num_data_curr)+'_horse_minloss.txt']);
        command_curr.extend(['-human_data_path',file_data_pre+str(num_data_curr)+'_face_noIm_minloss.txt']);
        command_curr.extend(['-numDecrease',str(1)]);
        command_curr.extend(['-tps_model_path',tps_model_path]);
        command_curr.extend(['-outDir',out_dir_pre+str(num_data_curr)]);
        command_curr.extend(['-bgr']);
        command_curr.extend(['-affine']);
        command_curr=' '.join(command_curr);
        commands.append(command_curr);

    commands=np.array(commands);
    commands_split=np.array_split(commands,num_scripts);
    for idx_commands,commands in enumerate(commands_split):
        out_file_script_curr=out_file_script+'_'+str(idx_commands)+'.sh';
        print idx_commands
        print out_file_script_curr
        print commands;
        util.writeFile(out_file_script_curr,commands);
  
def script_bl_affine_resume(system):
    # num_data=range(500,3500,500)+[3531];
    num_data=[3531,500,3000,1000,2500,1500,2000]
    file_data_pre='../data/small_datasets/matches_5_';
    if system=='vision4':
        out_dir_pre_pre='/data/maheen-data/horse_cvpr';
    elif system=='vision1':
        out_dir_pre_pre='/home/SSD3/maheen-data/horse_project'
    else:
        out_dir_pre_pre='/disk2/horse_cvpr'
    out_dir_pre=os.path.join(out_dir_pre_pre,'bl_affine_small_data/horse_')
    torch_file='train_full_model.th';
    out_file_script='../scripts/bl_affine_small_data_resume';
    tps_model_path='../models/affine_localization_net_untrained.dat'
    num_scripts=2;
    commands=[];
    epoch_size=56;

    for num_data_curr in num_data:
        command_curr=['th',torch_file];
        out_dir_in=out_dir_pre+str(num_data_curr);
        out_dir_out=os.path.join(out_dir_in,'resume_60');
        if num_data_curr<2000:
            command_curr.extend(['learningRate',str(1e-5)]);
        else:
            command_curr.extend(['learningRate',str(1e-4)]);
        command_curr.extend(['multiplierMid',str(1)]);
        command_curr.extend(['multiplierBottom',str(0.01)]);
        
        command_curr.extend(['-iterations',str(140*epoch_size)]);
        command_curr.extend(['-decreaseAfter',str(60*epoch_size)]);

        command_curr.extend(['-horse_data_path',file_data_pre+str(num_data_curr)+'_horse_minloss.txt']);
        command_curr.extend(['-human_data_path',file_data_pre+str(num_data_curr)+'_face_noIm_minloss.txt']);
        command_curr.extend(['-numDecrease',str(1)]);
        command_curr.extend(['-tps_model_path',tps_model_path]);

        command_curr.extend(['-outDir',out_dir_out]);

        command_curr.extend(['-bgr']);

        command_curr.extend(['-full_model_flag']);
        command_curr.extend(['-face_detection_model_path',os.path.join(out_dir_in,'intermediate','model_all_3360.dat')]);
        command_curr.extend(['-affine']);
        command_curr=' '.join(command_curr);
        commands.append(command_curr);

    commands=np.array(commands);
    commands_split=np.array_split(commands,num_scripts);
    for idx_commands,commands in enumerate(commands_split):
        out_file_script_curr=out_file_script+'_'+str(idx_commands)+'.sh';
        print idx_commands
        print out_file_script_curr
        print commands;
        util.writeFile(out_file_script_curr,commands);


def script_bl_tif_us():
    out_dir_meta='/home/SSD3/maheen-data/horse_project'
    out_dir_meta_meta=os.path.join(out_dir_meta,'tif_bl_us');
    util.mkdir(out_dir_meta_meta);
    
    train_animal_file=[os.path.join(out_dir_meta,'files_for_sheepCode/'),'_train_us_','.txt'];
    train_human_file=[os.path.join(out_dir_meta,'files_for_sheepCode/'),'_train_us_face_noIm.txt'];
    
    val_animal_file=[os.path.join(out_dir_meta,'files_for_sheepCode/'),'_test_us_','_minloss.txt'];
    val_human_file=[os.path.join(out_dir_meta,'files_for_sheepCode/'),'_test_us_face_noIm_minloss.txt'];

    exp_data=[('horse', 77,15),('sheep', 27,5)];
    torch_file_full='train_full_model.th';
    torch_file_warp='train_warping_net.th'
    out_file_script='../scripts/tif_bl_us.sh';
    commands=[]

    for animal_type,epoch_size_warp,epoch_size in exp_data:    
        out_dir_meta_curr=os.path.join(out_dir_meta_meta,animal_type);
        train_animal_curr=animal_type.join(train_animal_file);
        train_human_curr=animal_type.join(train_human_file);
        val_animal_curr=animal_type.join(val_animal_file);
        val_human_curr=animal_type.join(val_human_file);

        command_curr=['th',torch_file_warp];
        command_curr.extend(['-outDir',os.path.join(out_dir_meta_curr,'tps')]);
        command_curr.extend(['-horse_data_path',train_animal_curr]);
        command_curr.extend(['-human_data_path',train_human_curr]);
        command_curr.extend(['-model','../models/tps_localization_net_untrained.dat']);
        command_curr.extend(['-val_horse_data_path',val_animal_curr]);
        command_curr.extend(['-val_human_data_path',val_human_curr]);
        command_curr.extend(['-iterations',10*epoch_size_warp]);
        command_curr.extend(['-decreaseAfter',5*epoch_size_warp]);
        command_curr.extend(['-saveAfter',3*epoch_size_warp]);
        command_curr.extend(['-iterations_test',2]);
        command_curr.extend(['-batchSize_test',50]);
        command_curr=[str(c) for c in command_curr];
        command_curr=' '.join(command_curr);
        commands.append(command_curr);

        out_dir_meta_curr=os.path.join(out_dir_meta_meta,animal_type);
        train_animal_curr=animal_type.join(train_animal_file);
        train_human_curr=animal_type.join(train_human_file);
        val_animal_curr=animal_type.join(val_animal_file);

        command_curr=['th',torch_file_full];
        command_curr.extend(['learningRate',str(1e-2)]);
        command_curr.extend(['multiplierMid',str(0.1)]);
        command_curr.extend(['multiplierBottom',str(0.01)]);
        command_curr.extend(['-horse_data_path',train_animal_curr]);
        command_curr.extend(['-human_data_path',train_human_curr]);
        command_curr.extend(['-val_data_path',val_animal_curr]);
        command_curr.extend(['-numDecrease',str(2)]);
        command_curr.extend(['-tps_model_path',os.path.join(out_dir_meta_curr,'tps','final','model_all_final.dat')]);
        command_curr.extend(['-outDir',os.path.join(out_dir_meta_curr,'full_model')]);
        command_curr.extend(['-bgr']);
        command_curr.extend(['dual']);
        command_curr.extend(['-iterations',150*epoch_size]);
        command_curr.extend(['-saveAfter',30*epoch_size]);
        command_curr.extend(['-decreaseAfter',50*epoch_size]);
        command_curr=[str(c) for c in command_curr];
        command_curr=' '.join(command_curr);
        commands.append(command_curr);

    util.writeFile(out_file_script,commands);
    print out_file_script


def script_sheep_all(system):
    # out_dir_meta='/home/SSD3/maheen-data/horse_project'

    if system=='vision4':
        out_dir_meta='/data/maheen-data/horse_cvpr';
    elif system=='vision1':
        out_dir_meta='/home/SSD3/maheen-data/horse_project'
    else:
        out_dir_meta='/disk2/horse_cvpr'
    
    out_dir_meta_meta=os.path.join(out_dir_meta,'all_sheep_models');
    util.mkdir(out_dir_meta_meta);
    
    train_animal_file=[os.path.join(out_dir_meta,'data_check/'),'/matches_5_','_train_allKP.txt'];
    train_human_file=[os.path.join(out_dir_meta,'data_check/aflw/')+'matches_5_','_train_allKP_noIm.txt'];
    
    train_animal_file_minloss=[os.path.join(out_dir_meta,'data_check/'),'/matches_5_','_train_allKP_minloss.txt'];
    train_human_file_minloss=[os.path.join(out_dir_meta,'data_check/aflw/')+'matches_5_','_train_allKP_noIm_minloss.txt'];
    
    val_animal_file=[os.path.join(out_dir_meta,'data_check/'),'/matches_5_','_test_allKP_minloss.txt'];
    val_human_file=[os.path.join(out_dir_meta,'data_check/aflw/')+'matches_5_','_test_allKP_noIm_minloss.txt'];

    # exp_data=[('horse', 77,15),('sheep', 27,5)];
    # 432,7,34
    exp_data=[('sheep', 34,7)];
    torch_file_full='train_full_model.th';
    torch_file_warp='train_warping_net.th'
    torch_file_kp='train_kp_net.th'
    out_file_script='../scripts/all_sheep_us';
    commands=[]
    tps_model_path='../models/tps_localization_net_untrained_withTPS.dat'
    model_human_kp='../models/human_face_model.dat';
    model_kp_scratch='../models/vanilla_scratch_bn_large.dat'
    num_scripts=1;

    for animal_type,epoch_size_warp,epoch_size in exp_data:    
        out_dir_meta_curr=os.path.join(out_dir_meta_meta,animal_type);
        train_animal_curr=animal_type.join(train_animal_file);
        train_human_curr=animal_type.join(train_human_file);

        train_animal_curr_minloss=animal_type.join(train_animal_file_minloss);
        train_human_curr_minloss=animal_type.join(train_human_file_minloss);
        
        val_animal_curr=animal_type.join(val_animal_file);
        val_human_curr=animal_type.join(val_human_file);

        print (train_animal_curr);
        print (train_human_curr);

        print (train_animal_curr_minloss);
        print (train_human_curr_minloss);
        
        print (val_animal_curr);
        print (val_human_curr);
        if system!='vision1':
            all_files=[train_animal_curr,train_human_curr ,train_animal_curr_minloss,train_human_curr_minloss,val_animal_curr,val_human_curr]
            for file_curr in all_files:
                lines=util.readLinesFromFile(file_curr);
                lines=[line.replace('/home/SSD3/maheen-data/horse_project',out_dir_meta) for line in lines];
                util.writeFile(file_curr,lines);


        # command_curr=['th',torch_file_warp];
        # command_curr.extend(['-outDir',os.path.join(out_dir_meta_curr,'tps')]);
        # command_curr.extend(['-horse_data_path',train_animal_curr]);
        # command_curr.extend(['-human_data_path',train_human_curr]);
        # command_curr.extend(['-model','../models/tps_localization_net_untrained.dat']);
        # command_curr.extend(['-val_horse_data_path',val_animal_curr]);
        # command_curr.extend(['-val_human_data_path',val_human_curr]);
        # command_curr.extend(['-iterations',10*epoch_size_warp]);
        # command_curr.extend(['-decreaseAfter',5*epoch_size_warp]);
        # command_curr.extend(['-saveAfter',3*epoch_size_warp]);
        # command_curr.extend(['-iterations_test',2]);
        # command_curr.extend(['-batchSize_test',50]);
        # command_curr=[str(c) for c in command_curr];
        # command_curr=' '.join(command_curr);
        # commands.append(command_curr);

        # out_dir_meta_curr=os.path.join(out_dir_meta_meta,animal_type);
        # train_animal_curr=animal_type.join(train_animal_file);
        # train_human_curr=animal_type.join(train_human_file);
        # val_animal_curr=animal_type.join(val_animal_file);

        # command_curr=['th',torch_file_full];
        # command_curr.extend(['learningRate',str(1e-2)]);
        # command_curr.extend(['multiplierMid',str(0.1)]);
        # command_curr.extend(['multiplierBottom',str(0.01)]);
        # command_curr.extend(['-horse_data_path',train_animal_curr]);
        # command_curr.extend(['-human_data_path',train_human_curr]);
        # command_curr.extend(['-val_data_path',val_animal_curr]);
        # command_curr.extend(['-numDecrease',str(2)]);
        # command_curr.extend(['-tps_model_path',os.path.join(out_dir_meta_curr,'tps','final','model_all_final.dat')]);
        # command_curr.extend(['-outDir',os.path.join(out_dir_meta_curr,'full_model')]);
        # command_curr.extend(['-bgr']);
        # command_curr.extend(['dual']);
        # command_curr.extend(['-iterations',150*epoch_size]);
        # command_curr.extend(['-saveAfter',30*epoch_size]);
        # command_curr.extend(['-decreaseAfter',50*epoch_size]);
        # command_curr=[str(c) for c in command_curr];
        # command_curr=' '.join(command_curr);
        # commands.append(command_curr);

        command_curr=['th',torch_file_full];
        command_curr.extend(['learningRate',str(1e-3)]);
        command_curr.extend(['multiplierMid',str(0.1)]);
        command_curr.extend(['multiplierBottom',str(0.01)]);        
        command_curr.extend(['-iterations',str(200*epoch_size)]);
        command_curr.extend(['-decreaseAfter',str(100*epoch_size)]);
        command_curr.extend(['-saveAfter',30*epoch_size]);
        command_curr.extend(['-horse_data_path',train_animal_curr_minloss]);
        command_curr.extend(['-human_data_path',train_human_curr_minloss]);
        command_curr.extend(['-val_data_path',val_animal_curr]);
        command_curr.extend(['-numDecrease',str(1)]);
        command_curr.extend(['-tps_model_path',tps_model_path]);
        command_curr.extend(['-outDir',os.path.join(out_dir_meta_curr,'bl_tps_fix')]);
        command_curr.extend(['-bgr']);
        command_curr=[str(c) for c in command_curr];
        command_curr=' '.join(command_curr);
        commands.append(command_curr);

        # command_curr=['th',torch_file_kp];
        # command_curr.extend(['-model',model_human_kp]);
        # command_curr.extend(['-outDir',os.path.join(out_dir_meta_curr,'bl_ft')]);
        # command_curr.extend(['-mean_im_path','../data/aflw_cvpr_224_mean.png']);
        # command_curr.extend(['-std_im_path','../data/aflw_cvpr_224_std.png']);
        # command_curr.extend(['decreaseAfter',epoch_size*50]);
        # command_curr.extend(['-iterations',150*epoch_size]);
        # command_curr.extend(['-saveAfter',30*epoch_size]);
        # command_curr.extend(['-data_path',train_animal_curr_minloss]);
        # command_curr.extend(['-val_data_path',val_animal_curr]);
        # command_curr.extend(['-numDecrease',str(2)]);
        # command_curr=[str(c) for c in command_curr];
        # command_curr=' '.join(command_curr);
        # commands.append(command_curr);

        # command_curr=['th',torch_file_kp];
        # command_curr.extend(['-model',model_kp_scratch]);
        # command_curr.extend(['-outDir',os.path.join(out_dir_meta_curr,'bl_scratch')]);
        # command_curr.extend(['-mean_im_path','../data/sheep_224_mean.png']);
        # command_curr.extend(['-std_im_path','../data/sheep_224_std.png']);
        # command_curr.extend(['decreaseAfter',epoch_size*100]);
        # command_curr.extend(['-iterations',300*epoch_size]);
        # command_curr.extend(['-saveAfter',30*epoch_size]);
        # command_curr.extend(['-data_path',train_animal_curr_minloss]);
        # command_curr.extend(['-val_data_path',val_animal_curr]);
        # command_curr.extend(['-numDecrease',str(2)]);
        # command_curr=[str(c) for c in command_curr];
        # command_curr=' '.join(command_curr);
        # commands.append(command_curr);

        # command_curr=['th',torch_file_kp];
        # command_curr.extend(['-model',os.path.join(out_dir_meta_curr,'bl_scratch',\
        #     'intermediate','model_all_'+str(150*epoch_size)+'.dat')]);
        # command_curr.extend(['-outDir',os.path.join(out_dir_meta_curr,'bl_scratch',\
        #     'resume_150')]);
        # command_curr.extend(['-mean_im_path','../data/sheep_224_mean.png']);
        # command_curr.extend(['-std_im_path','../data/sheep_224_std.png']);
        # command_curr.extend(['decreaseAfter',epoch_size*100]);
        # command_curr.extend(['-iterations',100*epoch_size]);
        # command_curr.extend(['-saveAfter',30*epoch_size]);
        # command_curr.extend(['learningRate',1e-4]);
        # command_curr.extend(['-data_path',train_animal_curr_minloss]);
        # command_curr.extend(['-val_data_path',val_animal_curr]);
        # command_curr.extend(['-numDecrease',str(1)]);
        # command_curr=[str(c) for c in command_curr];
        # command_curr=' '.join(command_curr);
        # commands.append(command_curr);

        # command_curr=['th',torch_file_kp];
        # command_curr.extend(['-model',os.path.join(out_dir_meta_curr,'bl_scratch','resume_150','final','model_all_final.dat')]);
        # command_curr.extend(['-outDir',os.path.join(out_dir_meta_curr,'bl_scratch','resume_150','resume_for_50')]);
        # command_curr.extend(['-mean_im_path','../data/sheep_224_mean.png']);
        # command_curr.extend(['-std_im_path','../data/sheep_224_std.png']);
        # command_curr.extend(['decreaseAfter',epoch_size*150]);
        # command_curr.extend(['-iterations',50*epoch_size]);
        # command_curr.extend(['-saveAfter',30*epoch_size]);
        # command_curr.extend(['learningRate',1e-4]);
        # command_curr.extend(['-data_path',train_animal_curr_minloss]);
        # command_curr.extend(['-val_data_path',val_animal_curr]);
        # command_curr.extend(['-numDecrease',str(1)]);
        # command_curr=[str(c) for c in command_curr];
        # command_curr=' '.join(command_curr);
        # commands.append(command_curr);


    commands=np.array(commands);
    commands_split=np.array_split(commands,num_scripts);
    for idx_commands,commands in enumerate(commands_split):
        out_file_script_curr=out_file_script+'_'+str(idx_commands)+'.sh';
        print idx_commands
        print out_file_script_curr
        print commands;
        util.writeFile(out_file_script_curr,commands);

    util.writeFile(out_file_script,commands);
    print out_file_script


def makeAblationGraph():
    dir_meta='/home/SSD3/maheen-data/horse_project';
    click_str='http://vision1.idav.ucdavis.edu:1000'
    dir_server='/home/SSD3/maheen-data';
    out_dir=os.path.join(dir_meta,'iccv_img');
    util.mkdir(out_dir);
    out_dir=os.path.join(out_dir,'ablation_graphs');
    util.mkdir(out_dir);
    # to_include=['bl_ft','bl_alexnet','bl_tps','ours_affine','ours'];outside=True;
    to_include=['ours','bl_tps','bl_ft'];outside=False
    # to_include=['ours','bl_tps','bl_affine','bl_ft'];outside=False
    # to_include=['bl_ft','bl_alexnet','bl_tps','ours'];outside=True;
    out_file=os.path.join(out_dir,'_'.join(to_include)+'.pdf');

    num_data=range(500,3500,500)+[3531];
    bl_ft_dir=[os.path.join(dir_meta,'face_baselines_small_data_rerun_50_dec/matches_5_'),'_horse_minloss/resume_50/test_images'];
    bl_alexnet_dir=[os.path.join(dir_meta,'cvpr_rebuttal','imagenet_last_scratch_small_data/small_train_minloss_'),'/resume_50/test_images'];
    # bl_tps_dir=[os.path.join(dir_meta,'bl_tps_small_data_no_bn_fix_0.001_0.1_0.01/horse_'),'/test_images'];

    bl_tps_dir=[os.path.join(dir_meta,'bl_tps_lr_0.01_0.1_0.01/horse_'),'/test_images'];

    bl_affine_dir=[os.path.join(dir_meta,'bl_affine_small_data_0.001_0.1_0.01/horse_'),'/test_images'];

    us_affine_dir=[os.path.join(dir_meta,'full_nets_us_affine_small_data/horse_'),'/test_images'];
    us_tps_dir=[os.path.join(dir_meta,'full_nets_us_no_bn_fix/horse_'),'/test_images'];
    test_dirs_dict={'bl_ft':(bl_ft_dir,'g'),'bl_alexnet':(bl_alexnet_dir,'y'),'bl_tps':(bl_tps_dir,'r'),'bl_affine':(bl_affine_dir,'m'),\
                    'ours_affine':(us_affine_dir,'c'),'ours':(us_tps_dir,'b')};

    # test_dirs_dict = { key_curr: test_dirs_dict[key_curr] for key_curr in to_include }

    # bl_ft
    # /home/SSD3/maheen-data/horse_project/face_baselines_small_data_rerun_50_dec/matches_5_3531_horse_minloss/resume_50/test_images
    # bl_imagenet
    # /home/SSD3/maheen-data/horse_project/cvpr_rebuttal/imagenet_last_scratch_small_data/small_train_minloss_3531/resume_50/test_images
    # bl_tps
    # /home/SSD3/maheen-data/horse_project/bl_tps_small_data_no_bn_fix/horse_2000/resume_60
    # us_affine
    # /home/SSD3/maheen-data/horse_project/full_nets_us_affine_small_data/horse_1000/test_images/
    # us_tps
    # /home/SSD3/maheen-data/horse_project/full_nets_us_no_bn_fix/horse_1000/test_images/

    test_file='../data/test_minLoss_horse.txt';
    num_iter=2;
    batch_size=100;

    post_us=['_gt_pts.npy','_pred_pts.npy']
    
    # im_paths,gt_pt_files,pred_pt_files=viz.us_getFilePres(test_file,test_dir,post_us,num_iter,batch_size);
    colors=[];
    failures_all=[];
    labels=[];
    for key_curr in to_include:
        test_dir,color=test_dirs_dict[key_curr];
        failures_curr=[];
        for num_data_curr in num_data:
            test_dir_curr=str(num_data_curr).join(test_dir)
            errors_curr=viz.us_getErrorsAll(test_file,test_dir_curr,post_us,num_iter,batch_size);
            _,failures_kp=viz.getErrRates(errors_curr,0.1)
            failures_curr.append(failures_kp);

        failures_all.append(failures_curr);
        colors.append(color);
        labels.append(key_curr.replace('_',' ').upper());

    xAndYs=[(num_data,vals_curr) for vals_curr in failures_all];
    for val_curr in failures_all:
        print np.mean(np.array(val_curr)-np.array(failures_all[0]));
        print np.max(np.array(val_curr)-np.array(failures_all[0]));
        print val_curr;
    # print xAndYs
    xlabel='Training Data Size';
    ylabel='Failure %';
    
    visualize.plotSimple(xAndYs,out_file,xlabel=xlabel,ylabel=ylabel,legend_entries=labels,colors=colors,outside=outside);
    print out_file.replace(dir_server,click_str);

def makeComparisonBarGraphs():
    dir_meta='/home/SSD3/maheen-data/horse_project';
    dir_server='/home/SSD3/maheen-data';
    click_str='http://vision1.idav.ucdavis.edu:1000'
    out_dir=os.path.join(dir_meta,'iccv_img');
    util.mkdir(out_dir);
    out_dir=os.path.join(out_dir,'bar_graphs');
    util.mkdir(out_dir);
    for i in range(3,5):
    # i=6;

        if i==0:
            to_include=['bl_ft','bl_alexnet','bl_tps','ours_affine','ours'];limit=[0,40];outside=True;loc=None;post='horse';
        elif i==1:
            to_include=['ours','bl_ft','bl_tps'];outside=False;limit=[0,35];loc=2;post='horse';
        elif i==2:
            to_include=['ours','bl_ft','bl_tps'];outside=False;limit=[0,17];loc=2;post='sheep';
        elif i==3:
            to_include=['ours','tif'];outside=False;limit=[0,25];loc=6;post='horse_tif';
        elif i==4:
            to_include=['ours','tif'];outside=False;limit=[0,19];loc=6;post='sheep_tif';
        elif i==5:
            to_include=['ours','bl_tps','bl_ft','scratch'];limit=[0,40];outside=False;loc=6;post='horse';
        elif i==6:
            to_include=['ours','bl_tps','bl_ft','scratch'];limit=[0,45];outside=False;loc=2;post='sheep';
        elif i==7:
            to_include=['ours','ours_random'];outside=False;limit=[0,15];loc=2;post='horse';


    
        out_file=os.path.join(out_dir,'_'.join(to_include)+'_'+post+'.pdf');

        if post=='horse':
            num_data_curr=3531;
            bl_ft_dir=os.path.join(dir_meta,'face_baselines_small_data_rerun_50_dec/matches_5_3531_horse_minloss/resume_50/test_images');
            bl_alexnet_dir=os.path.join(dir_meta,'cvpr_rebuttal','imagenet_last_scratch_small_data/small_train_minloss_3531/resume_50/test_images');
            # bl_tps_dir=os.path.join(dir_meta,'bl_tps_small_data_no_bn_fix/horse_3531/resume_60/test_images');
            us_affine_dir=os.path.join(dir_meta,'full_nets_us_affine_small_data/horse_3531/test_images');
            # bl_tps_dir=os.path.join(dir_meta,'bl_tps_small_data_no_bn_fix_0.001_0.1_0.01/horse_3531/test_images');
            bl_tps_dir=os.path.join(dir_meta,'bl_tps_lr_0.01_0.1_0.01/horse_3531/resume_180_3/test_images');

            us_tps_dir=os.path.join(dir_meta,'full_nets_us_no_bn_fix/horse_3531/test_images');
            us_scratch_dir=os.path.join(dir_meta,'scratch_rebuttal','resume_150','test_images');
            bl_random=os.path.join(dir_meta,'train_random_neighbors_iccv','full_model','test_images');
            test_dirs_dict={'bl_ft':(bl_ft_dir,'g'),\
            'bl_alexnet':(bl_alexnet_dir,'y'),\
            'bl_tps':(bl_tps_dir,'r'),\
            'ours_affine':(us_affine_dir,'c'),\
            'ours':(us_tps_dir,'b'),\
            'scratch':(us_scratch_dir,'k'),\
            'ours_random':(bl_random,'g')};
            test_file='../data/test_minLoss_horse.txt';
            num_iter=2;
            batch_size=100;
        elif post=='sheep':
            bl_ft_dir=os.path.join(dir_meta,'all_sheep_models/sheep/bl_ft/test_images');
            bl_tps_dir=os.path.join(dir_meta,'all_sheep_models/sheep/bl_tps/test_images');# to change
            us_tps_dir=os.path.join(dir_meta,'all_sheep_models/sheep/full_model/test_images');
            us_scratch_dir=os.path.join(dir_meta,'all_sheep_models/sheep/bl_scratch/resume_150/test_images');
            test_dirs_dict={'bl_ft':(bl_ft_dir,'g'),'bl_tps':(bl_tps_dir,'r'),'ours':(us_tps_dir,'b'),'scratch':(us_scratch_dir,'k')};
            test_file=os.path.join(dir_meta,'data_check/sheep/matches_5_sheep_test_allKP_minloss.txt');
            num_iter=2;
            batch_size=50; 
        elif post=='horse_tif':   
            tif_dir=[os.path.join(dir_meta,'files_for_sheepCode','horse_test_new.txt'),\
            os.path.join(dir_meta,'files_for_sheepCode','horse_test_new_TIF_result.txt')];
            # them_test=[os.path.join(dir_meta,file_curr) for file_curr in them_test];
            # out_them=[file_curr[:file_curr.rindex('.')]+'_TIF_result.txt' for file_curr in them_test];
       

            us_tps_dir=os.path.join(dir_meta,'tif_bl_us/horse/full_model/test_images');
            test_dirs_dict={'tif':(tif_dir,'g'),'ours':(us_tps_dir,'b')};
            test_file=os.path.join(dir_meta,'files_for_sheepCode/horse_test_us_horse_minloss.txt');
            num_iter=2;
            batch_size=50; 
        elif post=='sheep_tif':   
            # tif_dir=os.path.join(dir_meta,'all_sheep_models/sheep/bl_ft/test_images');

            tif_dir=[os.path.join(dir_meta,'files_for_sheepCode','sheep_test_new.txt'),\
            os.path.join(dir_meta,'files_for_sheepCode','sheep_test_new_TIF_result.txt')];
            # them_test=[os.path.join(dir_input_data,file_curr) for file_curr in them_test];
            # out_them=[file_curr[:file_curr.rindex('.')]+'_TIF_result.txt' for file_curr in them_test];
       

            us_tps_dir=os.path.join(dir_meta,'tif_bl_us/sheep/full_model/test_images');
            test_dirs_dict={'tif':(tif_dir,'g'),'ours':(us_tps_dir,'b')};
            test_file=os.path.join(dir_meta,'files_for_sheepCode/sheep_test_us_sheep_minloss.txt');
            num_iter=2;
            batch_size=50;


        post_us=['_gt_pts.npy','_pred_pts.npy']
        ticks=['LE','RE','N','LM','RM','ALL'];
        # im_paths,gt_pt_files,pred_pt_files=viz.us_getFilePres(test_file,test_dir,post_us,num_iter,batch_size);
        colors=[];
        failures_all=[];
        labels=[];
        for key_curr in to_include:
            test_dir_curr,color=test_dirs_dict[key_curr];
            if key_curr=='tif':
                errors_curr=viz.them_getErrorsAll(test_dir_curr[0],test_dir_curr[1]);
            else:    
                errors_curr=viz.us_getErrorsAll(test_file,test_dir_curr,post_us,num_iter,batch_size);
        
            failures_all.append(errors_curr);
            colors.append(color);
            labels.append(key_curr.replace('_',' ').upper());

        err_rates_all=viz.plotComparisonKpError(failures_all,out_file,ticks,labels,colors=colors,ylim=limit,loc=loc);
        print err_rates_all[:,-1];
        print err_rates_all[:,-1]-err_rates_all[0,-1];
        print out_file.replace(dir_server,click_str);


def makeTifDirs():
    dir_meta='/home/SSD3/maheen-data/horse_project';
    dir_server='/home/SSD3/maheen-data';
    click_str='http://vision1.idav.ucdavis.edu:1000'
    out_dir=os.path.join(dir_meta,'iccv_img');
    util.mkdir(out_dir);
    out_dir_meta=os.path.join(out_dir,'tif_imgs');
    util.mkdir(out_dir_meta);
    for type_exp in ['horse','sheep']:
        if type_exp=='horse':
            tif_dir=[os.path.join(dir_meta,'files_for_sheepCode','horse_test_new.txt'),\
            os.path.join(dir_meta,'files_for_sheepCode','horse_test_new_TIF_result.txt')];
            out_tif_dir=os.path.join(out_dir_meta,type_exp);
            util.mkdir(out_tif_dir);
            batch_size=50;
            num_iter=2;
        elif type_exp=='sheep':
            tif_dir=[os.path.join(dir_meta,'files_for_sheepCode','sheep_test_new.txt'),\
            os.path.join(dir_meta,'files_for_sheepCode','sheep_test_new_TIF_result.txt')];
            out_tif_dir=os.path.join(out_dir_meta,type_exp);
            util.mkdir(out_tif_dir);
            batch_size=50;
            num_iter=2;

        gt_file_them=tif_dir[0];
        pred_file_them=tif_dir[1];

        im_paths,im_sizes,annos_gt=viz.readGTFile(gt_file_them);
        annos_pred=viz.readPredFile(pred_file_them);
        
        batch_num_curr=1;
        for i,(im_path,anno_gt,anno_pred) in enumerate(zip(im_paths,annos_gt,annos_pred)):
            pre=str(i/batch_size+1)+'_';
            i=i%batch_size;
            out_path=os.path.join(out_tif_dir,pre+str(i+1)+'_org.jpg');
            viz.saveImWithAnno(im_path,anno_pred,out_path)
            print out_path


def makeComparativeHTML():
    dir_meta='/home/SSD3/maheen-data/horse_project';
    dir_server='/home/SSD3/maheen-data';
    click_str='http://vision1.idav.ucdavis.edu:1000'
    out_dir=os.path.join(dir_meta,'iccv_img');
    out_dir_meta=os.path.join(out_dir,'comparative_htmls');
    util.mkdir(out_dir_meta);
    compare_whats=[['us_tps_horse','bl_tps_horse'],['us_tps_horse','bl_ft_horse']]
    # ,['us_tif_tps_sheep','tif_sheep']]
    # compare_whats=[['us_tps_sheep','bl_ft_sheep']]
    # ['us_tps_sheep','bl_tps_sheep']

    bl_ft_dir=os.path.join(dir_meta,'face_baselines_small_data_rerun_50_dec/matches_5_3531_horse_minloss/resume_50/test_images');
    bl_alexnet_dir=os.path.join(dir_meta,'cvpr_rebuttal','imagenet_last_scratch_small_data/small_train_minloss_3531/resume_50/test_images');
    us_affine_dir=os.path.join(dir_meta,'full_nets_us_affine_small_data/horse_3531/test_images');
    # bl_tps_dir=os.path.join(dir_meta,'bl_tps_small_data_no_bn_fix_0.001_0.1_0.01/horse_3531/test_images');
    bl_tps_dir=os.path.join(dir_meta,'bl_tps_lr_0.01_0.1_0.01/horse_3531/resume_180_3/test_images');
    us_tps_dir=os.path.join(dir_meta,'full_nets_us_no_bn_fix/horse_3531/test_images');
    us_scratch_dir=os.path.join(dir_meta,'scratch_rebuttal','resume_150','test_images');
    bl_ft_dir_sheep=os.path.join(dir_meta,'all_sheep_models/sheep/bl_ft/test_images');
    bl_tps_dir_sheep=os.path.join(dir_meta,'all_sheep_models/sheep/bl_tps/test_images');# to change
    us_tps_dir_sheep=os.path.join(dir_meta,'all_sheep_models/sheep/full_model/test_images');
    us_scratch_dir_sheep=os.path.join(dir_meta,'all_sheep_models/sheep/bl_scratch/resume_150/test_images');
    tif_dir=[os.path.join(dir_meta,'files_for_sheepCode','horse_test_new.txt'),\
          os.path.join(dir_meta,'files_for_sheepCode','horse_test_new_TIF_result.txt'),
          os.path.join(out_dir_meta,'tif_imgs','horse')];
    us_tif_tps_dir=os.path.join(dir_meta,'tif_bl_us/horse/full_model/test_images');
    tif_dir_sheep=[os.path.join(dir_meta,'files_for_sheepCode','sheep_test_new.txt'),\
          os.path.join(dir_meta,'files_for_sheepCode','sheep_test_new_TIF_result.txt'),\
          os.path.join(dir_meta,'iccv_img','tif_imgs','sheep')];
    us_tif_tps_dir_sheep=os.path.join(dir_meta,'tif_bl_us/sheep/full_model/test_images');
                

    dict_dirs={'bl_ft_horse':bl_ft_dir,\
    'bl_alexnet_horse':bl_alexnet_dir,\
    'us_affine_horse':us_affine_dir,\
    'bl_tps_horse':bl_tps_dir,\
    'us_tps_horse':us_tps_dir,\
    'us_scratch_horse':us_scratch_dir,\
    'bl_ft_sheep':bl_ft_dir_sheep,\
    'bl_tps_sheep':bl_tps_dir_sheep,\
    'us_tps_sheep':us_tps_dir_sheep,\
    'us_scratch_sheep':us_scratch_dir_sheep,\
    'tif_horse':tif_dir,\
    'us_tif_tps_horse':us_tif_tps_dir,\
    'tif_sheep':tif_dir_sheep,\
    'us_tif_tps_sheep':us_tif_tps_dir_sheep}
    
    post_ims_us=['_org_nokp.jpg','_gt.jpg','_warp_nokp.jpg','_warp.jpg','_org.jpg'];
    
    for compare_what in compare_whats:
        if 'tif' in compare_what[0]:
            if 'sheep' in compare_what[0]:
                test_file=os.path.join(dir_meta,'files_for_sheepCode/sheep_test_us_sheep_minloss.txt');
            elif 'horse' in compare_what[0]:
                test_file=os.path.join(dir_meta,'files_for_sheepCode/horse_test_us_horse_minloss.txt');
            batch_size=50;
            num_iter=2;
            post_ims_them=['_org.jpg'];
        else:
            if 'horse' in compare_what[0]:
                test_file='../data/test_minLoss_horse.txt';
                batch_size=100;
            elif 'sheep' in compare_what[0]:
                test_file=os.path.join(dir_meta,'data_check/sheep/matches_5_sheep_test_allKP_minloss.txt');
                batch_size=50;
            num_iter=2;
            if 'ft' in compare_what[1]:
                post_ims_them=['_org.jpg',];
            else:
                post_ims_them=['_warp_nokp.jpg','_warp.jpg','_org.jpg',];

        out_file_html=os.path.join(out_dir_meta,'_'.join(compare_what)+'.html');
        post_us=['_gt_pts.npy','_pred_pts.npy']
        them_dir=dict_dirs[compare_what[1]];
        us_dir=dict_dirs[compare_what[0]];
        keys=zip(compare_what,[us_dir,them_dir]);

        avgs=[];
        im_paths_all=[]
        idx_key=0;
        for key_curr,test_dir_curr in keys:
            
            # errors_curr=us_getErrorsAll(us_test,dir_curr,post_us,num_iter,batch_size);
            if key_curr.startswith('tif'):
                errors_curr=viz.them_getErrorsAll(test_dir_curr[0],test_dir_curr[1]);
                them_dir=test_dir_curr[2];
            else:
                # if idx_key==0:
                #     batch_size=50;
                #     num_iter=2;
                # else:
                batch_size=100;
                num_iter=1;
                _,gt_pt_files,_=viz.us_getFilePres(test_file,test_dir_curr,post_us,num_iter,batch_size);
                errors_curr=viz.us_getErrorsAll(test_file,test_dir_curr,post_us,num_iter,batch_size);
                im_paths_all.append(gt_pt_files);
            idx_key+=1;

            err=np.array(errors_curr);
            bin_keep=err>=0;
            err[err<0]=0;
            div=np.sum(bin_keep,1);
            sum_val=np.sum(err,1).astype(np.float);
            avg=sum_val/div;
            avgs.append(avg);
            
            
        # biggest_diff=avgs[1]-avgs[0];
        biggest_diff=avgs[0];
        idx_sort=np.argsort(biggest_diff)
        # [::-1];
        ims=[];
        captions=[];
        for idx_curr in idx_sort:
            file_curr=im_paths_all[0][idx_curr];
            file_curr=os.path.split(file_curr)[1];
            file_curr=file_curr[:file_curr.index('_gt')];
            files_us=[os.path.join(us_dir,file_curr+post_im_curr) for post_im_curr in post_ims_us ];
            captions_us=['us']*len(files_us);
            

            file_curr=file_curr.split('_');
            if int(file_curr[0])>1:
                # print file_curr
                file_curr[0]='1';
                file_curr[1]=str(int(file_curr[1])+50);
            # print file_curr    
            file_curr='_'.join(file_curr);
            

            files_them=[os.path.join(them_dir,file_curr+post_im_curr) for post_im_curr in post_ims_them ];
            captions_them=['them']*len(files_them);
            files_all=files_us+files_them;
            captions_all=captions_us+captions_them;
            files_all=[util.getRelPath(file_curr,dir_server) for file_curr in files_all];
            ims.append(files_all);
            captions.append(captions_all);
        
        visualize.writeHTML(out_file_html,ims,captions);
        print out_file_html.replace(dir_server,click_str);        
    
def makeThreshCurves():
    dir_meta='/home/SSD3/maheen-data/horse_project';
    dir_server='/home/SSD3/maheen-data';
    click_str='http://vision1.idav.ucdavis.edu:1000'
    out_dir=os.path.join(dir_meta,'iccv_img');
    util.mkdir(out_dir);
    out_dir=os.path.join(out_dir,'thresh_curves');
    util.mkdir(out_dir);
    # to_include=['bl_ft','bl_alexnet','bl_tps','ours_affine','ours'];outside=True;
    to_include=['ours','bl_tps','bl_ft','scratch'];post='horse';data_type='Horse';
    # to_include=['ours','bl_tps','bl_ft','scratch'];post='sheep';data_type='Sheep'
    # to_include=['ours','tif'];post='horse_tif';data_type='Horse';
    # to_include=['ours','tif'];post='sheep_tif';data_type='Sheep';
    to_include_all=[(['ours','bl_tps','bl_ft','scratch'],'horse','Horse'),(['ours','bl_tps','bl_ft','scratch'],'sheep','Sheep'),(['ours','tif'],'horse_tif','Horse'),(['ours','tif'],'sheep_tif','Sheep')];

    for to_include,post,data_type in to_include_all:
    # to_include=['ours','ours_random'];post='horse';data_type='Horse';

    
        out_file_pre=os.path.join(out_dir,'_'.join(to_include)+'_'+post+'_');
            # +'.pdf');
        if post=='horse':
            num_data_curr=3531;
            bl_ft_dir=os.path.join(dir_meta,'face_baselines_small_data_rerun_50_dec/matches_5_3531_horse_minloss/resume_50/test_images');
            bl_alexnet_dir=os.path.join(dir_meta,'cvpr_rebuttal','imagenet_last_scratch_small_data/small_train_minloss_3531/resume_50/test_images');
            # bl_tps_dir=os.path.join(dir_meta,'bl_tps_small_data_no_bn_fix_0.001_0.1_0.01/horse_3531/test_images');
            bl_tps_dir=os.path.join(dir_meta,'bl_tps_lr_0.01_0.1_0.01/horse_3531/resume_180_3/test_images');

            us_affine_dir=os.path.join(dir_meta,'full_nets_us_affine_small_data/horse_3531/test_images');
            us_scratch_dir=os.path.join(dir_meta,'scratch_rebuttal','resume_150','test_images');
            us_tps_dir=os.path.join(dir_meta,'full_nets_us_no_bn_fix/horse_3531/test_images');
            bl_random=os.path.join(dir_meta,'train_random_neighbors_iccv','full_model','test_images');
            test_dirs_dict={'bl_ft':(bl_ft_dir,'g'),'bl_alexnet':(bl_alexnet_dir,'y'),'bl_tps':(bl_tps_dir,'r'),'ours_affine':(us_affine_dir,'c'),'ours':(us_tps_dir,'b'),'scratch':(us_scratch_dir,'k'),\
            'ours_random':(bl_random,'g')};
            test_file='../data/test_minLoss_horse.txt';
            num_iter=2;
            batch_size=100;
        elif post=='sheep':
            bl_ft_dir=os.path.join(dir_meta,'all_sheep_models/sheep/bl_ft/test_images');
            bl_tps_dir=os.path.join(dir_meta,'all_sheep_models/sheep/bl_tps/test_images');
            us_tps_dir=os.path.join(dir_meta,'all_sheep_models/sheep/full_model/test_images');
            us_scratch_dir=os.path.join(dir_meta,'all_sheep_models/sheep/bl_scratch/resume_150/test_images');
            test_dirs_dict={'bl_ft':(bl_ft_dir,'g'),'bl_tps':(bl_tps_dir,'r'),'ours':(us_tps_dir,'b'),'scratch':(us_scratch_dir,'k')};
            test_file=os.path.join(dir_meta,'data_check/sheep/matches_5_sheep_test_allKP_minloss.txt');
            num_iter=2;
            batch_size=50; 
        elif post=='horse_tif':   
            tif_dir=[os.path.join(dir_meta,'files_for_sheepCode','horse_test_new.txt'),\
            os.path.join(dir_meta,'files_for_sheepCode','horse_test_new_TIF_result.txt')];
            us_tps_dir=os.path.join(dir_meta,'tif_bl_us/horse/full_model/test_images');
            test_dirs_dict={'tif':(tif_dir,'g'),'ours':(us_tps_dir,'b')};
            test_file=os.path.join(dir_meta,'files_for_sheepCode/horse_test_us_horse_minloss.txt');
            num_iter=2;
            batch_size=50; 
        elif post=='sheep_tif':   
            tif_dir=[os.path.join(dir_meta,'files_for_sheepCode','sheep_test_new.txt'),\
            os.path.join(dir_meta,'files_for_sheepCode','sheep_test_new_TIF_result.txt')];
            us_tps_dir=os.path.join(dir_meta,'tif_bl_us/sheep/full_model/test_images');
            test_dirs_dict={'tif':(tif_dir,'g'),'ours':(us_tps_dir,'b')};
            test_file=os.path.join(dir_meta,'files_for_sheepCode/sheep_test_us_sheep_minloss.txt');
            num_iter=2;
            batch_size=50;


        threshes=range(0,26);
        threshes=[float(thresh_curr)/100.0 for thresh_curr in threshes];
        post_us=['_gt_pts.npy','_pred_pts.npy']
        ticks=['LE','RE','N','LM','RM','ALL'];

        errors_all=[];
        curves=[];
        labels=[];
        colors=[];
        # for out_dir_test in dir_results:
        for key_curr in to_include:
            test_dir_curr,color=test_dirs_dict[key_curr];
            if key_curr=='tif':
                errors_curr=viz.them_getErrorsAll(test_dir_curr[0],test_dir_curr[1]);
            else:
                errors_curr=viz.us_getErrorsAll(test_file,test_dir_curr,post_us,num_iter,batch_size);

            curve_curr=[[],[],[],[],[],[]];
            for thresh_curr in threshes:
                failures,failures_kp=viz.getErrRates(errors_curr,thresh_curr)
                for failure_idx,failure_curr in enumerate([f_curr for f_curr in failures]+[failures_kp]):
                    curve_curr[failure_idx].append(failure_curr);
            curves.append(curve_curr);
            colors.append(color);
            labels.append(key_curr.replace('_',' ').upper());


        threshes_p=[thresh*100 for thresh in threshes]
        for curve_idx in range(6):
            out_file=out_file_pre+ticks[curve_idx]+'.pdf';
            xAndYs=[(threshes_p,curve_curr[curve_idx]) for curve_curr in curves];

            xlabel='Error Threshold %';
            ylabel='Failure Rate %'

            visualize.plotSimple(xAndYs,out_file,xlabel=xlabel,ylabel=ylabel,legend_entries=labels,title=data_type+' '+ticks[curve_idx],colors=colors);
            print out_file.replace(dir_server,click_str);

def rerun_scratch(system):
    if system=='vision4':
        out_dir_pre_pre='/data/maheen-data/horse_cvpr';
    elif system=='vision1':
        out_dir_pre_pre='/home/SSD3/maheen-data/horse_project'
    else:
        out_dir_pre_pre='/disk2/horse_cvpr'
    out_dir_pre=os.path.join(out_dir_pre_pre,'scratch_rebuttal')

    out_file_sh='../scripts/train_resume_scratch.sh';
    epoch_size=56;

    torch_file_kp='train_kp_net.th'
    commands=[];
    command_curr=['th',torch_file_kp];
    # command_curr.extend(['-model',os.path.join(out_dir_pre,'intermediate','model_all_'+str(56*180)+'.dat')]);
    # command_curr.extend(['-outDir',os.path.join(out_dir_pre,'resume_180')]);

    command_curr.extend(['-model',os.path.join(out_dir_pre,'resume_150','final','model_all_final.dat')]);
    command_curr.extend(['-outDir',os.path.join(out_dir_pre,'resume_150','resume_for_50')]);
    command_curr.extend(['-mean_im_path','../data/horse_mean.png']);
    command_curr.extend(['-std_im_path','../data/horse_std.png']);
    command_curr.extend(['decreaseAfter',epoch_size*150]);
    command_curr.extend(['-iterations',50*epoch_size]);
    command_curr.extend(['-saveAfter',30*epoch_size]);
    command_curr.extend(['learningRate',1e-4]);
    command_curr.extend(['-data_path','../data/train_horse_minloss.txt']);
    command_curr.extend(['-val_data_path','../data/test_minLoss_horse.txt']);
    command_curr.extend(['-numDecrease',str(1)]);
    command_curr=[str(c) for c in command_curr];
    command_curr=' '.join(command_curr);
    commands.append(command_curr);

    print out_file_sh
    util.writeFile(out_file_sh,commands);

def run_random_neighbors(vision1):
    # num_data=range(500,3500,500);
    num_data=[3531];
    if vision1:
        out_dir_pre_pre='/home/SSD3/maheen-data/horse_project'
    else:
        out_dir_pre_pre='/disk2/horse_cvpr'
    file_data_pre=os.path.join(out_dir_pre_pre,'neighbor_data/small_datasets/matches_random_5_')
    out_dir_pre=os.path.join(out_dir_pre_pre,'full_nets_us_no_bn_fix/horse_')
    out_dir_tps_meta=os.path.join(out_dir_pre_pre,'tps_nets_no_bn_fix/horse_')
    torch_file='train_full_model.th';
    torch_file_warp='train_warping_net.th';
    out_file_script='../scripts/train_random_neighbors';
    out_dir_meta_curr=os.path.join(out_dir_pre_pre,'train_random_neighbors_iccv');
    num_scripts=1;
    commands=[];
    epoch_size_warp=275;
    epoch_size=56;

    for num_data_curr in num_data:
        command_curr=['th',torch_file_warp];
        command_curr.extend(['-outDir',os.path.join(out_dir_meta_curr,'tps')]);
        command_curr.extend(['-horse_data_path',file_data_pre+str(num_data_curr)+'_horse.txt']);
        command_curr.extend(['-human_data_path',file_data_pre+str(num_data_curr)+'_face_noIm.txt']);
        command_curr.extend(['-model','../models/tps_localization_net_untrained.dat']);
        command_curr.extend(['-val_horse_data_path','../data/test_minLoss_horse.txt']);
        command_curr.extend(['-val_human_data_path','../data/test_minLoss_noIm_face.txt']);
        command_curr.extend(['-iterations',10*epoch_size_warp]);
        command_curr.extend(['-decreaseAfter',5*epoch_size_warp]);
        command_curr.extend(['-saveAfter',3*epoch_size_warp]);
        command_curr.extend(['-iterations_test',2]);
        command_curr.extend(['-batchSize_test',100]);
        command_curr=[str(c) for c in command_curr];
        command_curr=' '.join(command_curr);
        commands.append(command_curr);


        command_curr=['th',torch_file];
        command_curr.extend(['learningRate',str(1e-2)]);
        command_curr.extend(['multiplierMid',str(0.1)]);
        command_curr.extend(['multiplierBottom',str(0.01)]);
        command_curr.extend(['-horse_data_path',file_data_pre+str(num_data_curr)+'_horse.txt']);
        command_curr.extend(['-human_data_path',file_data_pre+str(num_data_curr)+'_face_noIm.txt']);
        command_curr.extend(['-numDecrease',str(2)]);
        command_curr.extend(['-tps_model_path',os.path.join(out_dir_meta_curr,'tps','final','model_all_final.dat')]);
        command_curr.extend(['-outDir',os.path.join(out_dir_meta_curr,'full_model')]);
        command_curr.extend(['-bgr']);
        command_curr.extend(['dual']);
        command_curr.extend(['-iterations',150*epoch_size]);
        command_curr.extend(['-saveAfter',30*epoch_size]);
        command_curr.extend(['-decreaseAfter',50*epoch_size]);
        command_curr=[str(c) for c in command_curr];
        command_curr=' '.join(command_curr);
        commands.append(command_curr);

    commands=np.array(commands);
    commands_split=np.array_split(commands,num_scripts);
    for idx_commands,commands in enumerate(commands_split):
        out_file_script_curr=out_file_script+'_'+str(idx_commands)+'.sh';
        print idx_commands
        print out_file_script_curr
        print commands;
        util.writeFile(out_file_script_curr,commands);

def run_vary_neighbors(system):
    # num_data=range(500,3500,500);
    num_data=[3531];
    num_neighbors=[2,3,4]
    # ,4];
    if system=='vision1':
        out_dir_pre_pre='/home/SSD3/maheen-data/horse_project'
    elif system=='vision4':
        out_dir_pre_pre='/data/maheen-data/horse_cvpr';
    else:
        out_dir_pre_pre='/disk2/horse_cvpr'
    
    # file_data_pre=os.path.join(out_dir_pre_pre,'neighbor_data/small_datasets/matches_random_5_')
    file_data_pre='../data/small_datasets/matches_'
    torch_file='train_full_model.th';
    torch_file_warp='train_warping_net.th';
    out_file_script='../scripts/train_neighbors_vary_1';
    out_dir_meta=os.path.join(out_dir_pre_pre,'train_neigbors_vary');
    util.mkdir(out_dir_meta);
    num_scripts=1;
    commands=[];
    epoch_size_warp=275;
    epoch_size=56;

    for num_neighbor in num_neighbors:
        for num_data_curr in num_data:
            out_dir_meta_curr=os.path.join(out_dir_meta,str(num_neighbor));
            util.mkdir(out_dir_meta_curr);
            # command_curr=['th',torch_file_warp];
            # command_curr.extend(['-outDir',os.path.join(out_dir_meta_curr,'tps')]);
            # command_curr.extend(['-horse_data_path',file_data_pre+str(num_neighbor)+'_'+str(num_data_curr)+'_horse.txt']);
            # command_curr.extend(['-human_data_path',file_data_pre+str(num_neighbor)+'_'+str(num_data_curr)+'_face_noIm.txt']);
            # command_curr.extend(['-model','../models/tps_localization_net_untrained.dat']);
            # command_curr.extend(['-val_horse_data_path','../data/test_minLoss_horse.txt']);
            # command_curr.extend(['-val_human_data_path','../data/test_minLoss_noIm_face.txt']);
            # command_curr.extend(['-iterations',10*epoch_size_warp]);
            # command_curr.extend(['-decreaseAfter',5*epoch_size_warp]);
            # command_curr.extend(['-saveAfter',3*epoch_size_warp]);
            # command_curr.extend(['-iterations_test',2]);
            # command_curr.extend(['-batchSize_test',100]);
            # command_curr=[str(c) for c in command_curr];
            # command_curr=' '.join(command_curr);
            # commands.append(command_curr);

            command_curr=['th',torch_file];
            command_curr.extend(['learningRate',str(1e-2)]);
            command_curr.extend(['multiplierMid',str(0.1)]);
            command_curr.extend(['multiplierBottom',str(0.01)]);
            command_curr.extend(['-horse_data_path',file_data_pre+str(num_neighbor)+'_'+str(num_data_curr)+'_horse.txt']);
            command_curr.extend(['-human_data_path',file_data_pre+str(num_neighbor)+'_'+str(num_data_curr)+'_face_noIm.txt']);
            command_curr.extend(['-numDecrease',str(2)]);
            command_curr.extend(['-tps_model_path',os.path.join(out_dir_meta_curr,'tps','final','model_all_final.dat')]);
            command_curr.extend(['-outDir',os.path.join(out_dir_meta_curr,'full_model')]);
            command_curr.extend(['-bgr']);
            command_curr.extend(['dual']);
            command_curr.extend(['-iterations',150*epoch_size]);
            command_curr.extend(['-saveAfter',30*epoch_size]);
            command_curr.extend(['-decreaseAfter',50*epoch_size]);
            command_curr=[str(c) for c in command_curr];
            command_curr=' '.join(command_curr);
            commands.append(command_curr);



    commands=np.array(commands);
    commands_split=np.array_split(commands,num_scripts);
    for idx_commands,commands in enumerate(commands_split):
        out_file_script_curr=out_file_script+'_'+str(idx_commands)+'.sh';
        print idx_commands
        print out_file_script_curr
        print commands;
        util.writeFile(out_file_script_curr,commands);


def getNN():
    dir_data='../data/small_datasets';
    in_file_pre='matches_5_3531_';
    
    animal_post='horse.txt';
    face_posts=['face.txt','face_noIm.txt'];

    for num_to_keep in [2,3,4]:
        out_file_pre='matches_'+str(num_to_keep)+'_3531_';
        

        data_in=util.readLinesFromFile(os.path.join(dir_data,in_file_pre+animal_post));
        data_in=np.array(data_in);
        _,data_in_uni_index=np.unique(data_in,return_index=True);
        data_in_uni=data_in[np.sort(data_in_uni_index)];
        idx_keep_all=[];

        for data_in_curr in data_in_uni:
            idx_curr=np.where(data_in==data_in_curr)[0];
            idx_curr=np.sort(idx_curr);
            idx_keep=idx_curr[:num_to_keep];
            # print idx_keep
            idx_keep_all.extend(list(idx_keep));
            # print idx_keep_all
            # raw_input();

        # _,data_in_uni_index=np.unique(data_in,return_index=True);
        # data_in_uni_index=np.sort(data_in_uni_index);

        for file_post in [animal_post]+face_posts:
            in_file_curr=os.path.join(dir_data,in_file_pre+file_post);
            out_file_curr=os.path.join(dir_data,out_file_pre+file_post);
            print in_file_curr,out_file_curr;
            data_in=util.readLinesFromFile(in_file_curr);
            data_keep=np.array(data_in)[idx_keep_all];
            print len(data_keep);


            util.writeFile(out_file_curr,data_keep);
    

    

def makeNeighborsGraph(system):
    num_data=[3531];
    num_neighbors=[1,5,10,15]
    dir_server='/home/SSD3/maheen-data';
    click_str='http://vision1.idav.ucdavis.edu:1000'
    if system=='vision1':
        out_dir_pre_pre='/home/SSD3/maheen-data/horse_project'
    elif system=='vision4':
        out_dir_pre_pre='/data/maheen-data/horse_cvpr';
    else:
        out_dir_pre_pre='/disk2/horse_cvpr'
    
    # file_data_pre=os.path.join(out_dir_pre_pre,'neighbor_data/small_datasets/matches_random_5_')
    out_file=os.path.join(out_dir_pre_pre,'iccv_img','nn_vary.pdf');
    out_dir_meta=os.path.join(out_dir_pre_pre,'train_neigbors_vary');
    test_file='../data/test_minLoss_horse.txt';
    post_us=['_gt_pts.npy','_pred_pts.npy']

    num_iter=2;
    batch_size=100;

    failures_curr=[];
    for num_neighbor in num_neighbors:
        test_dir_curr=os.path.join(out_dir_meta,str(num_neighbor),'full_model','test_images');
        errors_curr=viz.us_getErrorsAll(test_file,test_dir_curr,post_us,num_iter,batch_size);
        _,failures_kp=viz.getErrRates(errors_curr,0.1)
        failures_curr.append(failures_kp);

    xAndYs=[(num_neighbors,failures_curr)];
    # for val_curr in failures_all:
    #     print val_curr[0];
    # print xAndYs
    xlabel='Number of NN';
    ylabel='Failure %';
    outside=False;
    
    visualize.plotSimple(xAndYs,out_file,xlabel=xlabel,ylabel=ylabel,colors=['b'],outside=outside,xticks=num_neighbors);
    print out_file.replace(dir_server,click_str);

def main():
    # script_bl_tps_smooth(True)
    # script_bl_tps_lr_search()
    # script_bl_tps_lr_search()
    # script_bl_tps_resume(True)
    # makeComparisonBarGraphs()
    # makeNeighborsGraph('vision1');
    # script_bl_tps(True)
    # script_full_us_tps(True)
    # getNN()
    makeComparativeHTML()
    # makeTifDirs();
    # getNN();
    # script_bl_affine('vision3')
    # run_vary_neighbors('vision3')
    # copySmallDatatoVision3()
    # run_random_neighbors(True)
    # script_bl_tps_lr_search(vision1=False)
    # rerun_scratch('vision1');
    # makeThreshCurves();

    # makeAblationGraph()    
    # makeComparisonBarGraphs()

    # script_bl_tps_resume(vision1=True)
    # script_sheep_all('vision1');
    # script_bl_tps(False)
    # script_bl_affine('vision3')

    # script_bl_affine_resume('vision3');
    # script_affine_us('vision1')
    # script_full_us_affine('vision3');

    # script_bl_tps_resume(False);
    # script_full_us_tps(True);
    # script_tpsUs(True);
    # copySmallDatatoVision3();
    # rerun_5040_bl_ft();
    # rerun_5040_bl_alexnet()
    # copy_files_ext_disk();
    # '/home/SSD3/maheen-data/horse_project/face_baselines_small_data_rerun_50_dec'


if __name__=='__main__':
    main();

    