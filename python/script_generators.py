import util;
import numpy as np;
import os;
import shutil;
import itertools;

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
    num_neighbors=[str(num_curr) for num_curr in [5]];
    num_data=range(500,3500,500)+[3531];
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
    num_data=[3000,500,2500,1000,2000,1500]
    file_data_pre='../data/small_datasets/matches_5_';
    if vision1:
        out_dir_pre_pre='/home/SSD3/maheen-data/horse_project'
    else:
        out_dir_pre_pre='/disk2/horse_cvpr'
    out_dir_pre=os.path.join(out_dir_pre_pre,'tps_nets_bn_fix/horse_')
    torch_file='train_warping_net.th';
    out_file_script='../scripts/train_warping_small_data';
    num_scripts=2;
    commands=[];
    for num_data_curr in num_data:
        command_curr=['th',torch_file];
        command_curr.extend(['-outDir',out_dir_pre+str(num_data_curr)]);
        command_curr.extend(['-horse_data_path',file_data_pre+str(num_data_curr)+'_horse.txt']);
        command_curr.extend(['-human_data_path',file_data_pre+str(num_data_curr)+'_face_noIm.txt']);
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
    num_data=[3531]
    # ,500,2500,1000,2000,1500]
    file_data_pre='../data/small_datasets/matches_5_';
    if vision1:
        out_dir_pre_pre='/home/SSD3/maheen-data/horse_project'
    else:
        out_dir_pre_pre='/disk2/horse_cvpr'
    out_dir_pre=os.path.join(out_dir_pre_pre,'full_nets_us_bn_fix/horse_')
    out_dir_tps_meta=os.path.join(out_dir_pre_pre,'tps_nets_bn_fix/horse_')
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
        command_curr.extend(['-dual']);
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
    num_data=[3531,500,3000,1000,2500,1500,2000]
    file_data_pre='../data/small_datasets/matches_5_';
    if vision1:
        out_dir_pre_pre='/home/SSD3/maheen-data/horse_project'
    else:
        out_dir_pre_pre='/disk2/horse_cvpr'
    out_dir_pre=os.path.join(out_dir_pre_pre,'bl_tps_small_data/horse_')
    torch_file='train_full_model.th';
    out_file_script='../scripts/train_full_model_us_small_data';
    tps_model_path='../models/tps_localization_net_untrained_bn_fix_withTPS.dat'
    num_scripts=2;
    commands=[];
    epoch_size=56;

    for num_data_curr in num_data:
        command_curr=['th',torch_file];
        if num_data_curr<2000:
            command_curr.extend(['learningRate',str(1e-4)]);
        else:
            command_curr.extend(['learningRate',str(1e-3)]);
        command_curr.extend(['multiplierMid',str(1)]);
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
    


def main():
    script_bl_tps(True);
    # script_tpsUs(True);
    # copySmallDatatoVision3();
    # rerun_5040_bl_ft();
    # rerun_5040_bl_alexnet()
    # copy_files_ext_disk();
    # '/home/SSD3/maheen-data/horse_project/face_baselines_small_data_rerun_50_dec'


if __name__=='__main__':
    main();

    