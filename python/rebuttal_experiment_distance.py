import util;
import os;
import numpy as np;
import random;
import visualize;
def internal_pairwise_distance(kp):
    dist_arr=np.zeros((10,));
    dist_arr_idx=-1;
    for kp_idx_1 in range(kp.shape[0]-1):
        for kp_idx_2 in range(kp_idx_1+1,kp.shape[0]):
            dist_arr_idx=dist_arr_idx+1;
            # print kp_idx_1,kp_idx_2;
            if kp[kp_idx_1,2]<0 or kp[kp_idx_2,2]<0:
                # print ('should have skipped');
                continue;
            x1=kp[kp_idx_1,0];
            y1=kp[kp_idx_1,1];
            x2=kp[kp_idx_2,0];
            y2=kp[kp_idx_2,1];
            dist=np.sqrt((x1-x2)**2+(y1-y2)**2);
            # dist_arr_idx=kp_idx_1*kp.shape[0]+kp_idx_2
            # print dist_arr_idx;
            dist_arr[dist_arr_idx]=dist;
        # print dist_arr;
    return dist_arr;

def internal_distance_comparison(kp_animal_paths,kp_human_paths,avg_flag=False):

    dist_difference_all=-1*np.ones((len(kp_animal_paths),10));
    for idx_pair,(kp_animal_path_curr,kp_human_path_curr) in enumerate(zip(kp_animal_paths,kp_human_paths)):
        if idx_pair%500==0:
            print idx_pair;
        kp_animal=np.load(kp_animal_path_curr);
        kp_human=np.load(kp_human_path_curr);
        dist_animal=internal_pairwise_distance(kp_animal);
        dist_human=internal_pairwise_distance(kp_human);
    
        for idx_diff_diff,idx_diff in enumerate(range(dist_animal.shape[0])):
        # enumerate([0,1,4,7,8,9]):
            if dist_animal[idx_diff]==0 or dist_human[idx_diff]==0:
                continue;
            dist_difference_all[idx_pair,idx_diff_diff]=np.abs(dist_animal[idx_diff]-dist_human[idx_diff])

    # print dist_difference_all[0]
    sums=np.array(dist_difference_all);
    # print sums[0];
    sums[dist_difference_all<0]=0;
    # print sums[0];
    if avg_flag:
        sums=np.sum(sums,1);
    else:
        sums=np.sum(sums,0);
    # print sums

    # print dist_difference_all[0]
    counts=np.ones(dist_difference_all.shape);
    # print counts[0]
    counts[dist_difference_all<0]=0;
    # print counts[0]
    if avg_flag:
        counts=np.sum(counts,1);
    else:
        counts=np.sum(counts,0);
    
    # print counts;

    avg_dists=sums/counts;
    print avg_dists.shape;
    if avg_flag:
        avg_dists=np.mean(avg_dists);
    return avg_dists;
    # print dist_animal
    # print dist_human;
    # print dist_difference

def internal_avg_distance_comparison(kp_animal_paths,kp_human_paths):

    dist_diff_both=[];
    for kp_curr in [kp_animal_paths,kp_human_paths]:
        dist_diff_all=-1*np.ones((len(kp_curr),10));
        for idx_path_curr,kp_path_curr in enumerate(kp_curr):
            dist_diff_all[idx_path_curr,:]=internal_pairwise_distance(np.load(kp_path_curr));
        dist_diff_both.append(dist_diff_all);
    means_both=[];
    for dist_all_curr in dist_diff_both:
        sums=np.sum(dist_all_curr,0)
        counts=np.zeros(dist_all_curr.shape);
        counts[dist_all_curr>0]=1;
        counts=np.sum(counts,0);
        # print sums,counts
        mean_curr=sums/counts;
        means_both.append(mean_curr);

    diffs=np.abs(means_both[0]-means_both[1]);
    return diffs;



def animal_human_pairwise_distance(kp_animal,kp_human,center_flag=False):
    # print kp_animal
    # print kp_human
    assert np.all(kp_animal[:,2]>0);
    assert np.all(kp_human[:,2]>0);

    kp_animal=kp_animal[:,:2];
    kp_human=kp_human[:,:2];
    center_animal=np.sum(kp_animal,0)/kp_animal.shape[0];
    # kp_animal[0];
    # 
    
    center_human=np.sum(kp_human,0)/kp_human.shape[0];
    # kp_human[0];
    # 
    if center_flag:
        kp_animal_c=kp_animal-np.tile(center_animal[np.newaxis,:],(kp_animal.shape[0],1));
        kp_human_c=kp_human-np.tile(center_human[np.newaxis,:],(kp_human.shape[0],1));
    else:
        kp_animal_c=kp_animal
        kp_human_c=kp_human
    
    dists=np.power(np.sum(np.power(kp_animal_c-kp_human_c,2),1),0.5)
    return dists;

def script_gettingSimilarityStats():
    data_type_sheep='SHEEP';
    animal_data_sheep = "/home/SSD3/maheen-data/horse_project/files_for_sheepCode/sheep_train_us_sheep_minloss.txt"
    human_data_sheep = "/home/SSD3/maheen-data/horse_project/files_for_sheepCode/sheep_train_us_face_noIm_minloss.txt"

    data_type_horse='HORSE'
    animal_data_horse = "/home/SSD3/maheen-data/horse_project/files_for_sheepCode/horse_train_us_horse_minloss.txt"
    human_data_horse = "/home/SSD3/maheen-data/horse_project/files_for_sheepCode/horse_train_us_face_noIm_minloss.txt"

    tuples=[(data_type_horse,animal_data_horse,human_data_horse),(data_type_sheep,animal_data_sheep,human_data_sheep)]
    for data_type,animal_data,human_data in (tuples):
        print data_type
        kp_animal_paths=[line_curr.split(' ')[1] for line_curr in util.readLinesFromFile(animal_data)];
        kp_human_paths=[line_curr.split(' ')[0] for line_curr in util.readLinesFromFile(human_data)];

        print len(kp_animal_paths),len(kp_human_paths);

        # kp_human_paths_shuffle=kp_human_paths[:];
        # random.shuffle(kp_human_paths_shuffle);
        # avg_dists=internal_distance_comparison(kp_human_paths_shuffle,kp_human_paths);
        # print avg_dists;

        avg_dists=internal_avg_distance_comparison(kp_animal_paths,kp_human_paths);
        print avg_dists
        # print 'done';
        avg_dists=internal_distance_comparison(kp_animal_paths,kp_human_paths,False);
        print avg_dists;
        print avg_dists[[0,1,4,7,8,9]]/224*100
        print avg_dists/224*100


        dists_all=[];
        for kp_animal_path,kp_human_path in zip(kp_animal_paths,kp_human_paths):
            # print 'hello'
            kp_animal=np.load(kp_animal_path);
            # kp_animal=np.load(kp_human_paths[random.randint(0,len(kp_human_paths)-1)]);
            kp_human=np.load(kp_human_path);
            if np.any(kp_human[:,2]<0) or np.any(kp_animal[:,2]<0):
                continue;
            dist_curr=animal_human_pairwise_distance(kp_animal,kp_human);
            dists_all.append(list(dist_curr));
            # break;

        dists_all=np.array(dists_all);
        print dists_all.shape;
        print np.sum(dists_all,0)/dists_all.shape[0]
        print np.mean(dists_all);

        dists_all=[];
        for kp_animal_path,kp_human_path in zip(kp_animal_paths,kp_human_paths):
            # print 'hello'
            kp_animal=np.load(kp_animal_path);
            # kp_animal=np.load(kp_human_paths[random.randint(0,len(kp_human_paths)-1)]);
            kp_human=np.load(kp_human_path);
            if np.any(kp_human[:,2]<0) or np.any(kp_animal[:,2]<0):
                continue;
            dist_curr=animal_human_pairwise_distance(kp_animal,kp_human,True);
            dists_all.append(list(dist_curr));
            # break;

        dists_all=np.array(dists_all);
        print dists_all.shape;
        print np.sum(dists_all,0)/dists_all.shape[0]
        print np.mean(dists_all);

def getVisibilityCount(paths):
    counts=np.zeros((5,));
    for idx_path_curr,path_curr in enumerate(paths):
        if idx_path_curr%10000==0:
            print idx_path_curr;
        arr=np.load(path_curr);
        count_curr=np.sum(arr[:,2]>0);
        counts[count_curr-1]+=1;
    return counts;


def getCommandFull2LossTrain(path_to_th,out_dir,horse_data_path,human_data_path,tps_model_path,old_model):
    command=['th',path_to_th];
    command=command+['-outDir',out_dir];
    command=command+['-horse_data_path',horse_data_path];
    command=command+['-human_data_path',human_data_path];
    command=command+['-tps_model_path',tps_model_path];
    command=command+['-face_detection_model_path',old_model];
    command=' '.join(command);
    return command;

def getCommmandFull2LossTest(path_to_th_test,out_dir):
    command=['th',path_to_th_test];
    model_path=os.path.join(out_dir,'final','model_all_final.dat');
    out_dir_images=os.path.join(out_dir,'test_images_new');
    # command=command+['-outDir',out_dir];
    command=command+['-model_path',model_path];
    command=command+['-out_dir_images',out_dir_images];
    command=' '.join(command);

#     cmd:option('-model_path','/home/SSD3/maheen-data/horse_project/cvpr_rebuttal/full_model_bgr/final/model_all_final.dat');
#     -- '/home/SSD3/maheen-data/horse_project/cvpr_rebuttal/alexnet_fc_scratch_ft/final/model_all_final.dat');
# cmd:option('-out_dir_images','/home/SSD3/maheen-data/horse_project/cvpr_rebuttal/full_model_bgr/test_images');
    return command;    

def full_system_script():
    num_neighbors=[5];
    # range(5,10,5);
    num_data=range(500,3500,500);
    num_data=num_data+[3531]
    # num_data=[2000,1000,1500,500,3000,2500,3531]
    # num_data=[500,
    num_runs=2;
    # num_data=[3531];
    print num_data
    post_tags=['_horse.txt','_face.txt','_face_noIm.txt'];
    
    dir_neighbors='/home/SSD3/maheen-data/horse_project/neighbor_data';
    out_dir_breakdowns=os.path.join(dir_neighbors,'small_datasets');
    out_dir_meta_old='/home/SSD3/maheen-data/horse_project/full_system_small_data_eye_1e-2_10_100';
    out_dir_tps='/home/SSD3/maheen-data/horse_project/tps_small_data_1e-3_dec_5_eye';
    file_pre='matches';
    minloss_post='_minloss.txt';
    num_files=4;
    out_dir_meta='/home/SSD3/maheen-data/horse_project/full_system_small_data_tps_reruns';

    
    out_file_pres=[];
    for num_neighbor in num_neighbors:
        for num_data_curr in num_data:
            file_curr=file_pre+'_'+str(num_neighbor)+'_'+str(num_data_curr);
            out_file_pre=os.path.join(out_dir_breakdowns,file_curr);
            out_file_pres.append(out_file_pre);
    
    horse_train_data_paths=[file_curr+post_tags[0][:post_tags[0].rindex('.')]+minloss_post\
                            for file_curr in out_file_pres];
    human_train_data_paths=[file_curr+post_tags[2][:post_tags[2].rindex('.')]+minloss_post\
                            for file_curr in out_file_pres];
    dirs_tps=[os.path.split(file_curr)[1] for file_curr in out_file_pres];
    

    
#     out_dir_meta='/home/SSD3/maheen-data/horse_project/full_system_small_data_eye_1e-2_10_100_affine';
#     out_dir_tps='/home/SSD3/maheen-data/horse_project/affine_small_data_1e-3_dec_5_eye';
    
    util.mkdir(out_dir_meta);
    
    commands_all=[];
    
    path_to_th='/home/maheenrashid/Downloads/animal_human_kp/torch/train_full_model.th';
    path_to_th_test='/home/maheenrashid/Downloads/animal_human_kp/torch/test.th';
    path_to_py='/home/maheenrashid/Downloads/animal_human_kp/python/visualize_results.py';
    # '/home/maheenrashid/Downloads/horses/torch/train_full_system_2loss.th';
    out_file_sh=os.path.join(out_dir_meta,'commands_test_old')
    out_file_sh_py=os.path.join(out_dir_meta,'commands_test_py.sh')
    test_dirs_for_python=[];



    commands_all=[];
    for dir_tps,file_curr,file_human in zip(dirs_tps,horse_train_data_paths,human_train_data_paths):
        file_only=os.path.split(file_curr)[1];
        out_dir_old=os.path.join(out_dir_meta_old,file_only[:file_only.rindex('.')]);
        old_model=os.path.join(out_dir_old,'final','model_all_final.dat');
        tps_model_path=os.path.join(out_dir_tps,dir_tps,'final/model_all_final.dat');
        
        test_dirs_for_python.append(os.path.join(out_dir_old,'test_images_new'))
        command_test_curr=getCommmandFull2LossTest(path_to_th_test,out_dir_old);

        print tps_model_path,os.path.exists(tps_model_path);
        for idx_run in range(num_runs):
            out_dir=os.path.join(out_dir_meta,file_only[:file_only.rindex('.')]+'_'+str(idx_run));
            command_curr=getCommandFull2LossTrain(path_to_th,out_dir,file_curr,file_human,tps_model_path,old_model);
            
                            
            test_dirs_for_python.append(os.path.join(out_dir,'test_images'));
            
            assert os.path.exists(tps_model_path);
            assert os.path.exists(file_curr);
            assert os.path.exists(file_human);
            assert os.path.exists(old_model);
            print os.path.exists(tps_model_path), os.path.exists(file_curr), os.path.exists(file_human), os.path.exists(old_model);
            print out_dir
            
    #         print command_curr
            commands_all.append(command_test_curr);
            # commands_all_test.append(command_test_curr);

    commands_all=np.array(commands_all);
    commands_split=np.array_split(commands_all,num_files);
    for idx_commands_curr,commands_curr in enumerate(commands_split):
        out_file_sh_curr=out_file_sh+'_'+str(idx_commands_curr)+'.sh';
        # util.writeFile(out_file_sh_curr,commands_curr);

    python_commands=[];
    for py_out in test_dirs_for_python:
        command_curr=['python',path_to_py];
        command_curr.extend(['--test_dir',py_out]);
        command_curr=' '.join(command_curr);
        python_commands.append(command_curr);
    # util.writeFile(out_file_sh_py,python_commands);


    stat_files=[os.path.join(dir_curr,'stats.txt') for dir_curr in test_dirs_for_python];
    avgs=[];
    for file_curr in stat_files:
        avg_curr=float(util.readLinesFromFile(file_curr)[-1]);
        avgs.append(avg_curr);
    avgs=np.array(avgs);
    print avgs.shape;
    avgs=np.reshape(avgs,(avgs.shape[0]/3,3));
    avgs=avgs.T;
    avg_mean=np.mean(avgs,axis=0);

    print 'ok';
    print np.mean(avgs,axis=0);
    avgs=np.vstack((avgs,avg_mean[np.newaxis,:]));

    print avgs;
    
    # out_file_plot=os.path.join(cvpr_dir,'smoothing.png');
    # visualize.plotSimple(xAndYs
    cvpr_dir='/home/SSD3/maheen-data/horse_project/cvpr_figs'
    out_file=os.path.join(cvpr_dir,'smoothing.pdf');
    xAndYs=[(num_data,vals_curr) for vals_curr in avgs];
    # for val_curr in avg_failures:
    #     print val_curr[-1];
    print xAndYs
    xlabel='Training Data Size';
    ylabel='Failure %';
    labels=['Trial 1','Trial 2','Trial 3','Mean']
    visualize.plotSimple(xAndYs,out_file,xlabel=xlabel,ylabel=ylabel,legend_entries=labels);
    print out_file
    # print out_file.replace(dir_server,click_str);

#     range_data=range(0,len(commands_all),len(commands_all)/num_files);

#     if range_data[-1]!=len(commands_all):
# #         range_data[-1]=len(commands_all);
#         range_data.append(len(commands_all));
        
#     print range_data;
    
#     for num_file in range(num_files):
#         out_file_sh_curr=out_file_sh+'_'+str(num_file)+'.sh';
#         data_curr=commands_all[range_data[num_file]:range_data[num_file+1]];
        
#         util.writeFile(out_file_sh_curr,data_curr);
#         print out_file_sh_curr,len(data_curr);






def main():
    full_system_script();
    return

    # horse
    horse_data='../data/train_horse_minloss.txt';
    # human_data='../data/train_face_noIm_minloss.txt';

    # sheep
    sheep_data = '/home/SSD3/maheen-data/horse_project/data_check/sheep/matches_5_sheep_train_allKP_minloss.txt';
    # human_data = '/home/SSD3/maheen-data/horse_project/data_check/aflw/matches_5_sheep_train_allKP_noIm_minloss.txt';

    # face 
    face_data='/home/SSD3/maheen-data/data_face_network/aflw_cvpr_train.txt';
    
    data_types=['horse','sheep','face'];

    for data_type,file_curr in zip(data_types,[horse_data,sheep_data,face_data]):
        lines=util.readLinesFromFile(file_curr);
        paths=[line_curr.split(' ')[1] for line_curr in lines];
        print len(paths);
        counts=getVisibilityCount(paths);
        print data_type
        print counts;
        print counts/len(paths);



    # sheep
  #   [ 15.86894526  48.89542189  35.72347486  27.67282896  48.88523638
  # 28.24326921  35.25930504  25.82342399  28.05297121  30.46822719]

      # horse
  

    #[ 50.25351663  54.32783658  22.89629771  37.10918109  51.74207941  35.55950417  23.45348641  18.01694726  18.19582638  21.32575227]
    # [ 16.0628418   54.29982368  36.1488149   29.14258857  52.0141369   28.17827333  34.94214814  26.19167299  29.02118349  30.52509891]
    
    



if __name__=='__main__':
    main();