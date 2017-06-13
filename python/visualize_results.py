import util;
import os;
import numpy as np;
import visualize;
from optparse import OptionParser
import cv2;

def saveImWithAnno(im_path,anno,out_path):
    im=cv2.imread(im_path,1);
    label=anno;
    x=label[:,0];
    y=label[:,1];
    color=(0,0,255);
    colors=[(0,255,0)]*len(x)

    for label_idx in range(len(x)):
        cv2.circle(im,(x[label_idx],y[label_idx]),6,colors[label_idx],-1);
    print (im.shape)
    cv2.imwrite(out_path,im);
    
def parseAnnoStr(annos):
    annos=[int(num) for num in annos];
    annos=np.array(annos);
    annos=np.reshape(annos,(len(annos)/2,2));
    return annos;

def getDiffs(annos_gt,annos_pred):
    diffs_all=[];
    for anno_gt,anno_pred in zip(annos_gt,annos_pred):
        diffs=np.power(anno_gt-anno_pred,2);
        diffs=np.sum(diffs,1);
        diffs=np.power(diffs,0.5);
        diffs_all.append(diffs);
    return diffs_all;


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

def getErrRates(err,thresh=0.1):
    err=np.array(err);
    # print type(err);
    sum_errs=np.sum(err>thresh,0).astype(np.float);
    # print 'sum_errs',sum_errs;
    total_errs=np.sum(err>=0,0);
    err_rate=sum_errs/total_errs*100.0;
    sum_errs_tot=np.sum(sum_errs);
    total_errs_tot=np.sum(total_errs);
    err_rate_tot=sum_errs_tot/total_errs_tot*100.0;
    return err_rate,err_rate_tot;


def plotComparisonKpError(errors_all,out_file,ticks,labels,xlabel=None,ylabel=None,colors=None,thresh=0.1,\
                          title='',ylim=None,loc=None):
    vals={};
    err_check=np.array(errors_all);
    err_rates_all=[];
    for err in errors_all:
        err_rate,err_rate_tot=getErrRates(err);
        err_rate=[err_curr for err_curr in err_rate];
        if len(ticks)==len(err_rate)+1:
            err_rate.append(err_rate_tot);
        err_rates_all.append(err_rate);
    err_rates_all=np.array(err_rates_all);
    err_rates_all[err_rates_all==0]=0.1
    for idx_label_curr,label_curr in enumerate(labels):
        vals[label_curr]=err_rates_all[idx_label_curr,:];
    # print vals;

    if colors is None:
        colors=['b','g'];
        
    if xlabel is None:
        xlabel='Keypoint';
        
    if ylabel is None:
        ylabel='Failure Rate %';
 
        
    visualize.plotGroupBar(out_file,vals,ticks,labels,colors,xlabel=xlabel,ylabel=ylabel,\
                           width=1.0/len(vals),title=title,ylim=ylim,loc=loc);
    return err_rates_all

def readGTFile(file_curr):
    lines=util.readLinesFromFile(file_curr);
    im_paths=[];
    ims_size=[];
    annos_all=[];
    for line_curr in lines:
        line_split=line_curr.rsplit(None,20);
        im_paths.append(line_split[0]);
        
        im_size=line_split[1:1+4];
        im_size=[int(num) for num in im_size];
        im_size=[im_size[2]-im_size[0],im_size[3]-im_size[1]];
        ims_size.append(im_size);
        
        annos=line_split[1+4:];
        annos=parseAnnoStr(annos);
        annos_all.append(annos);
        
    return im_paths,ims_size,annos_all;
        
def readPredFile(pred_file):
    lines=util.readLinesFromFile(pred_file);
    annos_all=[];
    for line_curr in lines:
        annos_str=line_curr.split();
        annos=parseAnnoStr(annos_str);
        annos_all.append(annos);
    return annos_all;


def them_getErrorsAll(gt_file,pred_file):
    im_paths,im_sizes,annos_gt=readGTFile(gt_file);
    annos_pred=readPredFile(pred_file);
    diffs_all=getDiffs(annos_gt,annos_pred);
    errors_all=getErrorPercentageImSize(im_sizes,diffs_all);
    return errors_all;

def saveHTML(out_us,us_test,batch_size=50,num_iter=2,justHTML=False,eight=False):
    dir_server='./';
    post_us=['_gt_pts.npy','_pred_pts.npy']
    
    im_paths,gt_pt_files,pred_pt_files=us_getFilePres(us_test,out_us,post_us,num_iter,batch_size);
    if justHTML:
        post_ims_us=['_org_nokp.jpg','_gt.jpg','_warp_nokp.jpg','_warp.jpg','_org.jpg',];
        captions_for_row=['Input','Ground Truth','Warped Image','Prediction Warped','Prediction'];
        out_file_html=os.path.join(out_us,'results.html');
        
        idx_sort=range(len(gt_pt_files))
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
    else:
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

        labels=['Ours','thems'];
        if eight:
            ticks=['LTE','LBE','LE','RE','RBE','RTE','N','M','ALL'];
        else:
            ticks=['LE','RE','N','LM','RM','ALL'];
        colors=['b','g'];
        ylim=None;
        errors_all=[];

        errors_curr=us_getErrorsAll(us_test,out_us,post_us,num_iter,batch_size);
        failures,failures_kp=getErrRates(errors_curr,0.1)
        errors_all.append(errors_curr)
        errors_all.append(errors_curr[:])

        out_file_kp_err=os.path.join(out_us,'bar.pdf');
        err_rates_all=plotComparisonKpError(errors_all,out_file_kp_err,ticks,labels,colors=colors,ylim=ylim);
        out_file_stats=os.path.join(out_us,'stats.txt');
        print err_rates_all;
        string=[str(num_curr) for num_curr in err_rates_all[0]];
        print string
        # print failures,failures_kp
        # print errors_all
        # string=' '.join(string);
        util.writeFile(out_file_stats,string);




if __name__=='__main__':

    parser = OptionParser()
    parser.add_option("--test_dir",
                  action="store", type="string",help="dir with test output. specified as out_dir_images in test.th")
    parser.add_option("--test_file",action="store", default="../data/test_minLoss_horse.txt",type="string",help="test data file. specified as val_data_path in test.th")
    parser.add_option('--batchSize',type=int,default=100,help="batchSize specified in test.th");
    parser.add_option('--iterations',type=int,default=2,help="iterations specified in test.th");
    parser.add_option('--justHTML',action="store_true",default=False,help="does not require per im score files");
    parser.add_option('--eight',action="store_true",default=False,help="for 8 keypoints");

    (options, args) = parser.parse_args()
    # print options;
    # print args;
    # out_us='/home/SSD3/maheen-data/horse_project/test_git/test_full_trained_model/test_images';
    # us_test='../data/test_minLoss_horse.txt';
    saveHTML(options.test_dir,options.test_file,options.batchSize,options.iterations,options.justHTML,options.eight)

    