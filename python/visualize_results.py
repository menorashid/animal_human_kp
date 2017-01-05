import util;
import os;
import numpy as np;
import visualize;
from optparse import OptionParser


def us_getDiffs(gt_pt_files,pred_pt_files):
    diffs_all=[];
    for gt_file,pred_file in zip(gt_pt_files,pred_pt_files):
        gt_pts=np.load(gt_file);
        pred_pts=np.load(pred_file);
        bin_keep=gt_pts[:,2]>0
        diffs_curr=-1*np.ones((gt_pts.shape[0],));
        gt_pts=gt_pts[bin_keep,:2];
        pred_pts=pred_pts[bin_keep,:2];
        diffs=np.power(gt_pts-pred_pts,2);
        diffs=np.sum(diffs,1);
        diffs=np.power(diffs,0.5);
        diffs_curr[bin_keep]=diffs;
        diffs_all.append(diffs_curr);
    return diffs_all;


def getErrorPercentageImSize(im_sizes,diffs_all):
    errors_all=[];
    for im_size,diffs in zip(im_sizes,diffs_all):
        assert (im_size[0]==im_size[1]);
        errors=-1*np.ones(diffs.shape);
        errors[diffs>0]=diffs[diffs>0]/im_size[0];
        errors_all.append(errors);
    return errors_all;

def us_getFilePres(gt_file,out_dir_us,post_us,num_iter,batch_us):
    files_gt=[];
    files_pred=[];
    im_paths=util.readLinesFromFile(gt_file);
    im_paths=[im_path[:im_path.index(' ')] for im_path in im_paths];
    num_gt=len(im_paths);
    count=0;
    for batch_num in range(num_iter):
        for im_num in range(batch_us):
            file_pre=str(batch_num+1)+'_'+str(im_num+1);
            file_gt=file_pre+post_us[0];
            file_pred=file_pre+post_us[1];
            files_gt.append(os.path.join(out_dir_us,file_gt));
            files_pred.append(os.path.join(out_dir_us,file_pred));
    files_gt=files_gt[:num_gt];
    files_pred=files_pred[:num_gt];
    return im_paths,files_gt,files_pred;

def us_getErrorsAll(gt_file,out_dir_us,post_us,num_iter,batch_size):
    im_paths,gt_pt_files,pred_pt_files=us_getFilePres(gt_file,out_dir_us,post_us,num_iter,batch_size);
    diffs_all=us_getDiffs(gt_pt_files,pred_pt_files);
    im_sizes=[[2.0,2.0]]*len(diffs_all)
    errors_all=getErrorPercentageImSize(im_sizes,diffs_all);
    return errors_all;

def saveHTML(out_us,us_test,batch_size=50,num_iter=2):
    dir_server='./';
    post_us=['_gt_pts.npy','_pred_pts.npy']
    
    im_paths,gt_pt_files,pred_pt_files=us_getFilePres(us_test,out_us,post_us,num_iter,batch_size);
    errors_curr=us_getErrorsAll(us_test,out_us,post_us,num_iter,batch_size);
    err=np.array(errors_curr);
    bin_keep=err>=0;
    err[err<0]=0;
    div=np.sum(bin_keep,1);
    sum_val=np.sum(err,1).astype(np.float);
    avg=sum_val/div;

    post_ims_us=['_org_nokp.jpg','_gt.jpg','_warp_nokp.jpg','_warp.jpg','_org.jpg',];
    captions_for_row=['Input','Ground Truth','Warped Image','Prediction Warped','Prediction'];
    out_file_html=os.path.join(out_us,'results.html');
    idx_sort=np.argsort(avg)
    ims=[];
    captions=[];
    for idx_idx,idx_curr in enumerate(idx_sort):
        file_curr=gt_pt_files[idx_curr];
        file_curr=os.path.split(file_curr)[1];
        file_curr=file_curr[:file_curr.index('_gt')];
        files_us=[os.path.join(dir_server,file_curr+post_im_curr) for post_im_curr in post_ims_us ];
        captions_us=[str(idx_idx)+' '+caption_curr for caption_curr in captions_for_row];
        ims.append(files_us);
        captions.append(captions_us);
    
    visualize.writeHTML(out_file_html,ims,captions);
    print out_file_html


if __name__=='__main__':

    parser = OptionParser()
    parser.add_option("--test_dir",
                  action="store", type="string",help="dir with test output. specified as out_dir_images in test.th")
    parser.add_option("--test_file",action="store", default="../data/test_minLoss_horse.txt",type="string",help="test data file. specified as val_data_path in test.th")
    parser.add_option('--batchSize',type=int,default=50,help="batchSize specified in test.th");
    parser.add_option('--iterations',type=int,default=2,help="iterations specified in test.th");

    (options, args) = parser.parse_args()
    # print options;
    # print args;
    # out_us='/home/SSD3/maheen-data/horse_project/test_git/test_full_trained_model/test_images';
    # us_test='../data/test_minLoss_horse.txt';
    saveHTML(options.test_dir,options.test_file,options.batchSize,options.iterations)

    