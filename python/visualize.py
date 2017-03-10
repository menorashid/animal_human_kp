import matplotlib
import numpy as np;
matplotlib.use('Agg')
# matplotlib.use('PS') 
import matplotlib.pyplot as plt;
matplotlib.rcParams.update({'font.size': 16})
from matplotlib.backends.backend_pdf import PdfPages
import os;
from PIL import Image,ImageDraw,ImageFont;
import scipy.misc
import util;

def writeHTML(file_name,im_paths,captions,height=200,width=200):
    f=open(file_name,'w');
    html=[];
    f.write('<!DOCTYPE html>\n');
    f.write('<html><body>\n');
    f.write('<table>\n');
    for row in range(len(im_paths)):
        f.write('<tr>\n');
        for col in range(len(im_paths[row])):
            f.write('<td>');
            f.write(captions[row][col]);
            f.write('</td>');
            f.write('    ');
        f.write('\n</tr>\n');

        f.write('<tr>\n');
        for col in range(len(im_paths[row])):
            f.write('<td><img src="');
            f.write(im_paths[row][col]);
            # f.write('" height=100%; width=100%;"/></td>');
            f.write('" height='+str(height)+' width='+str(width)+'"/></td>');
            f.write('    ');
        f.write('\n</tr>\n');
        f.write('<p></p>');
    f.write('</table>\n');
    f.close();

def getHeatMap(arr,max_val=255):
    cmap = plt.get_cmap('jet')
    rgba_img = cmap(arr)
    rgb_img = np.delete(rgba_img, 3, 2)
    rgb_img = rgb_img*max_val
    # print rgb_img.shape,rgba_img.shape
    return rgb_img

def fuseAndSave(img,heatmap,alpha,out_file_curr=None):
    im=(img*alpha)+(heatmap*(1-alpha));

    if out_file_curr is None:
        return im;
    else:
        scipy.misc.imsave(out_file_curr,im);


def visualizeFlo(flo,file_name_x,file_name_y):
    plt.figure();plt.imshow(flo[:,:,0]);plt.savefig(file_name_x);
    plt.close();
    plt.figure();plt.imshow(flo[:,:,1]);plt.savefig(file_name_y);
    plt.close();

def createScatterOfDiffsAndDistances(diffs,title,xlabel,ylabel,out_file,dists=None):
    plt.figure();
    
    

    print out_file

    diffs_all=diffs.ravel();
    dists_all=[];
    if dists is None:
        dists_all=np.arange(diffs.shape[1]);
        dists_all=np.repeat(dists_all,diffs.shape[0]);
        # for idx in range(len(diffs)):
        #     dists_all.extend(range(1,len(diffs[idx])));
    else:
        dists_all=np.ravel(dists)
        # for dist in dists:
        #     dists_all.extend(dist);
    # bins=(max(diffs_all)-min(diffs_all),max(diffs_all)-min(diffs_all));
    heatmap, xedges, yedges = np.histogram2d(dists_all,diffs_all,bins=(100,45))
    heatmap=heatmap.T
    print heatmap.shape
    # print bins

    # extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    
    plt.imshow(heatmap)
    plt.savefig(out_file);
    plt.close();


def saveMatAsImage(mat,out_file):
    plt.figure();
    plt.imshow(mat);
    plt.savefig(out_file);
    plt.close();


def createImageAndCaptionGrid(img_paths,gt_labels,indices,text_labels):
    im_paths=[[[] for i in range(indices.shape[1]+1)] for j in range(indices.shape[0])]
    captions=[[[] for i in range(indices.shape[1]+1)] for j in range(indices.shape[0])]
    for r in range(indices.shape[0]):
        im_paths[r][0]=img_paths[r];
        captions[r][0]='GT class \n'+text_labels[gt_labels[r]]+' '+str(gt_labels[r]);
        for c in range(indices.shape[1]):
            pred_idx=indices[r][c]
            im_paths[r][c+1]=img_paths[pred_idx];
            if gt_labels[pred_idx] !=gt_labels[r]:
                captions[r][c+1]='wrong \n'+text_labels[gt_labels[pred_idx]]+' '+str(gt_labels[pred_idx]);
            else:
                captions[r][c+1]='';
    return im_paths,captions


def plotDistanceHistograms(diffs_curr,degree,out_file,title='',xlabel='Distance Rank',ylabel='Frequency',delta=0,dists_curr=None,bins=10,normed=False):
    
    if dists_curr is None:
        dists_curr=np.array(range(1,diffs_curr.shape[1]+1));
        dists_curr=np.expand_dims(dists_curr,0);
        dists_curr=np.repeat(dists_curr,diffs_curr.shape[0],0);

    # diffs_to_analyze=[0,45,90,135,180];
    # plt.ion();
    # for diff_to_analyze in diffs_to_analyze:
    diffs=diffs_curr-degree;
    diffs=abs(diffs);
    idx=np.where(diffs<=delta)
    dists=dists_curr[idx[0],idx[1]];

    plt.figure();
    print  'len(dists)',len(dists);
    plt.hist(dists,bins,normed=normed);
    plt.title(title);
    plt.xlabel(xlabel);
    plt.ylabel(ylabel);
    plt.savefig(out_file);
    plt.close();

def hist(dists,out_file,bins=10,normed=True,xlabel='Value',ylabel='Frequency',title='',cumulative=False):
    plt.figure();
    plt.hist(dists,bins,normed=normed,cumulative=cumulative);
    plt.title(title);
    plt.xlabel(xlabel);
    plt.ylabel(ylabel);
    plt.savefig(out_file);
    plt.close();    

def plotErrorBars(dict_to_plot,x_lim,y_lim,xlabel,y_label,title,out_file,margin=[0.05,0.05],loc=2):
    
    plt.title(title);
    plt.xlabel(xlabel);
    plt.ylabel(y_label);
    
    if y_lim is None:
        y_lim=[1*float('Inf'),-1*float('Inf')];
    
    max_val_seen_y=y_lim[1]-margin[1];
    min_val_seen_y=y_lim[0]+margin[1];
    print min_val_seen_y,max_val_seen_y
    max_val_seen_x=x_lim[1]-margin[0];
    min_val_seen_x=x_lim[0]+margin[0];
    handles=[];
    for k in dict_to_plot:
        means,stds,x_vals=dict_to_plot[k];
        
        min_val_seen_y=min(min(np.array(means)-np.array(stds)),min_val_seen_y);
        max_val_seen_y=max(max(np.array(means)+np.array(stds)),max_val_seen_y);
        
        min_val_seen_x=min(min(x_vals),min_val_seen_x);
        max_val_seen_x=max(max(x_vals),max_val_seen_x);
        
        handle=plt.errorbar(x_vals,means,yerr=stds);
        handles.append(handle);
        print max_val_seen_y
    plt.xlim([min_val_seen_x-margin[0],max_val_seen_x+margin[0]]);
    plt.ylim([min_val_seen_y-margin[1],max_val_seen_y+margin[1]]);
    plt.legend(handles, dict_to_plot.keys(),loc=loc)
    plt.savefig(out_file);
    plt.close();

def plotSimple(xAndYs,out_file,title='',xlabel='',ylabel='',legend_entries=None,loc=0,outside=False,logscale=False,colors=None,xticks=None):
    plt.title(title);
    plt.grid(1);
    plt.xlabel(xlabel);
    plt.ylabel(ylabel);
    if logscale:
        plt.gca().set_xscale('log')
    # assert len(xs)==len(ys)
    handles=[];
    for idx_x_y,(x,y) in enumerate(xAndYs):
        if colors is not None:
            color_curr=colors[idx_x_y];
            handle,=plt.plot(x,y,color_curr,linewidth=5.0);
        else:
            handle,=plt.plot(x,y,linewidth=5.0);

        handles.append(handle);
    if legend_entries is not None:
        if outside:
            lgd=plt.legend(handles,legend_entries,loc=loc,bbox_to_anchor=(1.05, 1),borderaxespad=0.)
        else:
            lgd=plt.legend(handles,legend_entries,loc=loc)    

    if xticks is not None:
        ax = plt.gca()
        ax.set_xticks(xticks)

    if legend_entries is not None:
        plt.savefig(out_file,bbox_extra_artists=(lgd,), bbox_inches='tight')
    else:
        plt.savefig(out_file);

    plt.close();    

def writeHTMLForFolder(path_to_im,ext='jpg',height=300,width=300):
    im_files=[file_curr for file_curr in os.listdir(path_to_im) if file_curr.endswith(ext)];
    im_files.sort();
    im_paths=[[im_file_curr] for im_file_curr in im_files];
    captions=im_paths;
    out_file_html=os.path.join(path_to_im,path_to_im[path_to_im.rindex('/')+1:]+'.html');
    writeHTML(out_file_html,im_paths,captions,height=height,width=width);


def plotGroupBar(out_file,dict_vals,xtick_labels,legend_vals,colors,xlabel='',ylabel='',title='',width=0.25,ylim=None,loc=None):
    # print 'loc',loc
    if loc is None:
        loc=2;
    # Setting the positions and width for the bars
    # if ylim is None:
    #     vals=dict_vals.values();
    #     vals=[v for v in val for val in values];
    #     ylim[
    plt.figure();
    plt.grid(1);
    plt.title(title);
    plt.xlabel(xlabel);
    plt.ylabel(ylabel);
    # w=len(legend_vals)*width
    # pos = np.arange(0,w*len(xtick_labels),w)
    pos=range(len(xtick_labels));

    pos = [pos_curr+(pos_curr*width) for pos_curr in pos]
    # pos[0]=0.0;
    # print pos
    # width = 0.25

    # Plotting the bars
    # fig, ax = plt.subplots(figsize=(10,5))


    # Create a bar with pre_score data,
    # in position pos,

    for pos_idx,legend_val in enumerate(legend_vals):
        # print legend_val,[p + width*pos_idx for p in pos],dict_vals[legend_val]
        # print [p + width*pos_idx for p in pos],dict_vals[legend_val],width,colors[pos_idx],legend_val
        plt.bar([p + width*pos_idx for p in pos],dict_vals[legend_val],width,color=colors[pos_idx],label=legend_val)

    ax = plt.gca()
    
    ax.set_xticks([p + len(legend_vals)/2.0 * width for p in pos])
    # print 'xticks' ,[p + len(legend_vals)/2.0 * width for p in pos]
    ax.set_xticklabels(xtick_labels,rotation=0)
    # ax.legend( legend_vals,loc=loc)
    if loc>5:
        plt.legend(legend_vals,bbox_to_anchor=(0., 0, 1., 1), ncol=2);
    else:
        ax.legend( legend_vals,loc=loc)
        # , mode="expand", borderaxespad=0.)
# Setting the x-axis and y-axis limits
    # plt.xlim(xlim[0],xlim[1])
    if ylim is not None:
        plt.ylim(ylim )

# Adding the legend and showing the plot
    
    # plt.gcf().subplots_adjust(top=0.9,bottom=0.15)
    # if out_file.endswith('.pdf'):
    #     with PdfPages(out_file) as pdf:
    #         pdf.savefig()
    # else:
    plt.savefig(out_file,bbox_inches='tight');
    plt.close();  

def plotBBox(img_path,bboxes,out_file,colors=None,labels=None):

    if type(img_path)==type('str'):
        im=scipy.misc.imread(img_path);
    else:
        im=img_path;

    if len(im.shape)<3:
        im=np.dstack((im,im,im));    
    im=np.asarray(im,np.uint8)
    im=Image.fromarray(im)
    
    if im.mode != "RGB":
        im.convert("RGB")
    # print im.size
    # print im.mode

    if labels is not None:
        assert len(labels)==len(bboxes);
        fontsize=max(20,im.size[0]/40);
        font = ImageFont.truetype("/usr/share/fonts/truetype/msttcorefonts/arial.ttf", fontsize)
        

    draw = ImageDraw.Draw(im)
    
    for idx_bbox,bbox in enumerate(bboxes):
        bbox_curr=[bbox[1],bbox[0],bbox[3],bbox[2]];
        # print bbox_curr
        if colors is None:
            color_curr=(255,255,255);
        elif type(colors)==type((1,2)):
            color_curr=colors;
        else:
            color_curr=colors[idx_bbox]
        # print color_curr;
        # print bbox_curr
        bbox_curr=[int(val) for val in bbox_curr]
        draw.rectangle(bbox_curr,outline=color_curr);

        if labels is not None:
            label_curr= labels[idx_bbox];
            if type(label_curr)!=type('str'):
                label_curr= '%.2f' %label_curr
            # print labels    
            color_curr= (color_curr[0],color_curr[1],color_curr[2],255);
            draw.text((bbox_curr[0],bbox_curr[1]), label_curr, font=font,fill=color_curr)

    im.save(out_file);


def writeHTMLForDifferentFolders(out_file_html,folders,captions,img_names,rel_path_replace=None,height=200,width=200):
    if rel_path_replace is None:
        string_curr=folders[0];
        rel_path_replace=string_curr[:string_curr[1:].index('/')+1];

    # print rel_path_replace

    img_paths=[];
    captions_html=[];
    
    for img_name in img_names:
        captions_row = [];
        img_paths_row = [];
        for caption_curr,folder in zip(captions,folders):
            img_paths_row.append(util.getRelPath(os.path.join(folder,img_name),rel_path_replace));
            captions_row.append(caption_curr);
        img_paths.append(img_paths_row);
        captions_html.append(captions_row);


    writeHTML(out_file_html,img_paths,captions_html,height=height,width=width);

def plotImageAndAnno(im_file,out_file,anno,color_curr=(0,255,0)):
    im=scipy.misc.imread(im_file);
    im=Image.fromarray(im)
    if im.mode != "RGB":
        im.convert("RGB")
    draw = ImageDraw.Draw(im)
    anno=list(anno.ravel());
    draw.point(anno,fill=color_curr)
    im.save(out_file);



def main():
    print 'hello';

if __name__=='__main__':
    main();