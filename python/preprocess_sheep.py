import os;
import util;
import visualize;
import numpy as np;
from collections import Counter;
import cv2;
import preprocessing_data;
dir_server='/home/SSD3/maheen-data/';
click_str='http://vision1.idav.ucdavis.edu:1000/';

def mergeFiles(org_files,org_data_replace=None):
    org_data=[];
    for file_curr in org_files:
        if 'set' in file_curr:
            lines_curr=modifySetFile(file_curr);
        else:
            lines_curr=util.readLinesFromFile(file_curr);
        org_data=org_data+lines_curr;

    if org_data_replace is not None:
        org_data=[line_curr.replace(org_data_replace[0],org_data_replace[1]) for line_curr in org_data];
    return org_data;

def compileAnnoFiles(in_folder,ext='.txt'):
    in_files_all=[os.path.join(root,file_curr) for root,dirs,files in os.walk(in_folder) for file_curr in files if file_curr.endswith(ext)]
    in_files_all=[file_curr for file_curr in in_files_all if not os.path.split(file_curr)[0].endswith('_annotations')];
    in_files_all=[file_curr for file_curr in in_files_all if 'set' in file_curr and os.path.split(file_curr)[1]!='dataset_list.txt'];
    return in_files_all;

def modifySetFile(in_file):
    lines=util.readLinesFromFile(in_file);
    lines=[line_curr for line_curr in lines if '.jpg' in line_curr]
    lines=[line_curr.strip('"') for line_curr in lines];
    lines=[' '.join(line_curr.split(',')) for line_curr in lines];
    return lines;

def getDataExists(org_data,num_tokens=20):
    org_data=list(set(org_data));
    
    ims_org=[line_curr.rsplit(' ',num_tokens)[0] for line_curr in org_data];
    
    not_exist=[];
    for idx,im_curr in enumerate(ims_org):
        if not os.path.exists(im_curr):
            im_curr_new=os.path.join(os.path.split(im_curr)[0],os.path.split(im_curr)[1].replace('%','_'));
            if os.path.exists(im_curr_new):
                line_curr=org_data[idx];
                line_curr_new=line_curr.replace(im_curr,im_curr_new);
                org_data[idx]=line_curr_new;
                ims_org[idx]=im_curr_new;
            else:
                not_exist.append(idx);
    
    not_exist.sort();
    not_exist=not_exist[::-1];
    
    for idx_rem in not_exist:
        del ims_org[idx_rem];
        del org_data[idx_rem];

    org_data_split=[line_curr.rsplit(' ',num_tokens) for line_curr in org_data];
    return org_data,org_data_split;

def getMappingBetweenLists(org_data_split,new_data_split):
    ims_anno=[[' '.join(line_curr[:2]) for line_curr in lines] for lines in [org_data_split,new_data_split] ]
    ims_overlap=np.intersect1d(ims_anno[0],ims_anno[1]);
    lines_match=[];
    for im_overlap in ims_overlap:
        idx_org=ims_anno[0].index(im_overlap);
        idx_new=ims_anno[1].index(im_overlap);
        lines_match.append((org_data_split[idx_org],new_data_split[idx_new]));

    print 'len(lines_match)',len(lines_match)
    ims_ex_org=np.setdiff1d(ims_anno[0],ims_anno[1]);
    ims_ex_new=np.setdiff1d(ims_anno[1],ims_anno[0]);
    print len(ims_ex_org),len(ims_ex_new);
    just_ims=[line_curr[0] for line_curr in org_data_split] 
    not_found=[];
    for idx_im_ex,im_ex in enumerate(ims_ex_org):
        idx_org=ims_anno[0].index(im_ex);
        line_org=org_data_split[idx_org];
        
        box_1=[int(val) for val in line_org[1:5]];
        box_1=[box_1[0],box_1[0]+box_1[2],box_1[1],box_1[1]+box_1[3]];

        im_curr=just_ims[idx_org];
        assert im_ex.startswith(im_curr);
        new_candidates=[line for line in ims_ex_new if line.startswith(im_curr[:im_curr.rindex('.')])];

        found=False;
        for candidate in new_candidates:
            idx_new=ims_anno[1].index(candidate);
            line_new=new_data_split[idx_new];
            box_2=[int(val) for val in line_new[1:5]];
            if box_1==box_2:
                lines_match.append((line_org,line_new));
                found=True;
                break;  

        if not found:
            not_found.append(line_org);

    print 'len(lines_match)',len(lines_match),len(not_found)

    # check boxes;
    for line_org,line_new in lines_match:
        box_1=[int(val) for val in line_org[1:5]];
        box_2=[int(val) for val in line_new[1:5]];
        box_1=[box_1[0],box_1[0]+box_1[2],box_1[1],box_1[1]+box_1[3]];
        assert box_1==box_2;

    return lines_match,not_found;

def visualizePoints(im_path,out_file,points):
    im=cv2.imread(im_path);
    for point in points:
        if point[2]>0:
            cv2.circle(im,(int(point[0]),int(point[1])),6,(255,0,0),-1);
    cv2.imwrite(out_file,im);


def getBBoxAndNumpyOrg(org_line):
    # print org_line
    box_1=[int(val) for val in org_line[1:5]];
    box_1=[box_1[0],box_1[0]+box_1[2],box_1[1],box_1[1]+box_1[3]];
    nums=org_line[5:];
    nums=np.array([int(num) for num in nums]);
    nums=np.reshape(nums,(len(nums)/2,2));
    nums=np.concatenate((nums,np.ones((nums.shape[0],1))),1);
    return box_1,nums;


def saveImAndNumpyNoExclusion():
    dir_meta='../data'
    dir_data=os.path.join(dir_meta,'sheep_org');

    out_dir_meta=os.path.join(dir_meta,'sheep_for_eight');
    util.mkdir(out_dir_meta);

    raw_anno_tif_file=os.path.join(out_dir_meta,'raw_anno_tif.txt')
    raw_anno_us_file=os.path.join(out_dir_meta,'raw_anno_us.txt');

    org_data_replace=['./',dir_data+'/'];   
    new_data_replace=['/home/laoreja/data/sheep/',dir_data+'/'];

    org_files=[os.path.join(dir_data,file_curr+'.txt') for file_curr in ['train','test']]+compileAnnoFiles(dir_data);
    new_files=[os.path.join(dir_data,file_curr+'.txt') for file_curr in ['sheep']];
    
    org_data=mergeFiles(org_files,org_data_replace);
    new_data=mergeFiles(new_files,new_data_replace);
    
    org_data,org_data_split = getDataExists(org_data,20);
    new_data,new_data_split = getDataExists(new_data,19);

    lines_match,not_found=getMappingBetweenLists(org_data_split,new_data_split);

    # fix image paths for multiple sheep in one image
    for line_org,line_new in lines_match:
        line_org[0]=line_new[0];

    # get matched data
    new_data=[' '.join(line_curr) for _,line_curr in lines_match];
    new_data_split=[line_curr for _,line_curr in lines_match];
    org_data=[' '.join(line_curr) for line_curr,_ in lines_match];
    org_data_split=[line_curr for line_curr,_ in lines_match];

    print org_data[0],new_data[0]
    util.writeFile(raw_anno_tif_file,org_data);
    
    

    params_dict={};
    params_dict['path_txt'] = raw_anno_tif_file;
    params_dict['path_pre'] = None;
    params_dict['type_data'] = 'sheep_tif';
    params_dict['out_dir_meta'] = out_dir_meta
    params_dict['out_dir_im'] = os.path.join(params_dict['out_dir_meta'],'im');
    params_dict['out_dir_npy'] = os.path.join(params_dict['out_dir_meta'],'npy_8');
    params_dict['out_file_list_npy'] = os.path.join(params_dict['out_dir_npy'],'data_list.txt');
    params_dict['out_file_list_im'] = os.path.join(params_dict['out_dir_im'],'data_list.txt');
    params_dict['out_file_pairs'] = os.path.join(params_dict['out_dir_meta'],'pairs_8.txt');
    params_dict['out_file_problem']=os.path.join(params_dict['out_dir_meta'],'problem_8.txt');
    params_dict['overwrite'] = True;
    params_dict['resize']=(224,224);
    params_dict['buff_ratio']=None;
    params=preprocessing_data.createParams('makeBboxPairFiles');
    params=params(**params_dict);
    path_im,bbox,anno_points=preprocessing_data.script_makeBboxPairFiles(params)


    for idx_line_new,new_data_line in enumerate(new_data_split):
        path_im_curr=path_im[idx_line_new];
        assert new_data_line[0].startswith(path_im_curr[:path_im_curr.rindex('.')]);
        bbox_curr=[str(int(val)) for val in bbox[idx_line_new]]
        bbox_line=new_data_line[1:5];
        if bbox_curr!=bbox_line:
            print bbox_line,bbox_curr
            new_data_line[1:5]=bbox_curr[:];
            

    print 'ROUND 2';#sanity check
    for idx_line_new,new_data_line in enumerate(new_data_split):
        path_im_curr=path_im[idx_line_new];

        assert new_data_line[0].startswith(path_im_curr[:path_im_curr.rindex('.')]);
        bbox_curr=[str(int(val)) for val in bbox[idx_line_new]]
        bbox_line=new_data_line[1:5];
        if bbox_curr!=bbox_line:
            print bbox_line,bbox_curr
            new_data_line[1:5]=bbox_curr[:];
            
    new_data=[' '.join(line_curr) for line_curr in new_data_split];
    util.writeFile(raw_anno_us_file,new_data);
    print raw_anno_us_file

    params_dict={};
    params_dict['path_txt'] = raw_anno_us_file;
    params_dict['path_pre'] = None;
    params_dict['type_data'] = 'sheep';
    params_dict['out_dir_meta'] = out_dir_meta
    params_dict['out_dir_im'] = os.path.join(params_dict['out_dir_meta'],'im');
    params_dict['out_dir_npy'] = os.path.join(params_dict['out_dir_meta'],'npy');
    params_dict['out_file_list_npy'] = os.path.join(params_dict['out_dir_npy'],'data_list.txt');
    params_dict['out_file_list_im'] = os.path.join(params_dict['out_dir_im'],'data_list.txt');
    params_dict['out_file_pairs'] = os.path.join(params_dict['out_dir_meta'],'pairs.txt');
    params_dict['out_file_problem']=os.path.join(params_dict['out_dir_meta'],'problem.txt');
    params_dict['overwrite'] = True;
    params_dict['resize']=(224,224);
    params_dict['buff_ratio']=None;
    params=preprocessing_data.createParams('makeBboxPairFiles');
    params=params(**params_dict);
    preprocessing_data.script_makeBboxPairFiles(params)

def script_saveDataAndVerify():

    saveImAndNumpyNoExclusion()

    dir_meta='../data'
    out_dir_meta=os.path.join(dir_meta,'sheep_for_eight');
    out_dir_im=os.path.join(dir_server,'horse_project','scratch/sheep_viz_noEx');
    util.mkdir(out_dir_im);
    out_file_html=os.path.join(out_dir_im,'sanity_check.html');

    ims_html=[];
    captions_html=[];
    for out_pre,file_curr in [('org',os.path.join(out_dir_meta,'pairs.txt')),('new',os.path.join(out_dir_meta,'pairs_8.txt'))]:
        row_html=[];
        caption_html=[];
        for idx_line,line_curr in enumerate(util.readLinesFromFile(file_curr)):
            line_curr = line_curr.split(' ');
            im=line_curr[0];
            npy=line_curr[1];
            out_file=os.path.join(out_dir_im,out_pre+'_'+str(idx_line)+'.jpg');
            visualizePoints(im,out_file,np.load(npy));
            row_html.append(util.getRelPath(out_file,dir_server));
            caption_html.append(' ');
        ims_html.append(row_html);
        captions_html.append(caption_html);
    ims_html=np.array(ims_html).T
    captions_html=np.array(captions_html).T
    visualize.writeHTML(out_file_html,ims_html,captions_html);
    print out_file_html.replace(dir_server,click_str);

def script_makeTrainTestFiles():
    dir_meta='../data';
    dir_old=os.path.join(dir_meta,'sheep');
    dir_new=os.path.join(dir_meta,'sheep_for_eight');
    files=[file_curr for file_curr in os.listdir(dir_old) if file_curr.startswith('matches_5') and file_curr.endswith('.txt')]
    replace_string=['/home/SSD3/maheen-data/horse_project/data_check/sheep',dir_new];
    replace_string_8=[os.path.join(dir_new,'npy'),os.path.join(dir_new,'npy_8')];
    
    out_files=[];
    for file_curr in files:
        old_file=os.path.join(dir_old,file_curr);
        out_file=os.path.join(dir_new,file_curr);
        lines=util.readLinesFromFile(old_file);
        lines=[line_curr.replace(replace_string[0],replace_string[1]) for line_curr in lines];
        # print old_file,out_file,lines[0];
        out_files.append(out_file)
        util.writeFile(out_file,lines);

    out_files_eight=[];
    for file_curr in out_files:
        out_file=file_curr[:file_curr.rindex('.')]+'_8.txt';
        lines=util.readLinesFromFile(file_curr);
        lines=[line_curr.replace(replace_string_8[0],replace_string_8[1]) for line_curr in lines];
        # print out_file,lines[0];
        out_files_eight.append(out_file);
        util.writeFile(out_file,lines);

    for file_curr in out_files+out_files_eight:

        lines=util.readLinesFromFile(file_curr);
        print file_curr,len(lines);
        
        for line_curr in lines:
            line_split=line_curr.split(' ');
            for split_curr in line_split:
                assert os.path.exists(split_curr)


def main():
    pass;
    

if __name__=='__main__':
    main();
