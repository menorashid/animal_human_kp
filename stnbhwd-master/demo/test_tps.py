import numpy as np;
import scipy.io;
import os;
import matplotlib.pyplot as plt;    
def _U_func_numpy(x1, y1, x2, y2):
    # Return zero if same point
    if x1 == x2 and y1 == y2:
        return 0.

    # Calculate the squared Euclidean norm (r^2)
    r_2 = (x2 - x1) ** 2 + (y2 - y1) ** 2

    # Return the squared norm (r^2 * log r^2)
    return r_2 * np.log(r_2)


def stage_1():
    num_control_points=16;
    grid_size = np.sqrt(num_control_points)
    x_control_source, y_control_source = np.meshgrid(
        np.linspace(-1, 1, grid_size),
        np.linspace(-1, 1, grid_size))

    # Create 2 x num_points array of source points
    source_points = np.vstack(
            (y_control_source.flatten(), x_control_source.flatten()))

    print source_points;

    # Get number of equations
    num_equations = num_control_points + 3


    L=np.zeros((num_equations,num_equations),dtype=np.float);
    L[0, 3:num_equations] = 1.
    L[1:3, 3:num_equations] = source_points
    L[3:num_equations, 0] = 1.
    L[3:num_equations, 1:3] = source_points.T

    # Loop through each pair of points and create the K matrix
    for point_1 in range(num_control_points):
        for point_2 in range(point_1, num_control_points):

            L[point_1 + 3, point_2 + 3] = _U_func_numpy(
                    source_points[0, point_1], source_points[1, point_1],
                    source_points[0, point_2], source_points[1, point_2])

            if point_1 != point_2:
                L[point_2 + 3, point_1 + 3] = L[point_1 + 3, point_2 + 3]

    # Invert
    L_inv = np.linalg.inv(L);
    # print L
    # print L_inv
    L_th=np.load('L.npy');
    L_inv_th=np.load('L_inv.npy');
    print np.max(np.abs(L-L_th));
    # print L[np.where(np.abs(L-L_th)>0.5)];
    print np.max(np.abs(L_inv-L_inv_th));

    out_height = 20
    # np.array(height / downsample_factor[0]).astype('int64')
    out_width = 20
    # np.array(width / downsample_factor[1]).astype('int64')
    x_t, y_t = np.meshgrid(np.linspace(-1, 1, out_width),
                        np.linspace(-1, 1, out_height))
    ones = np.ones(np.prod(x_t.shape))
    orig_grid = np.vstack([y_t.flatten(), x_t.flatten(), ones])
    orig_grid = orig_grid[0:2, :]
    # orig_grid = orig_grid.astype(theano.config.floatX)
    print orig_grid.shape
    print orig_grid[:,:10];

    # Construct right mat

    # First Calculate the U function for the new point and each source
    # point as in ref [2]
    # The U function is simply U(r) = r^2 * log(r^2), where r^2 is the
    # squared distance
    to_transform = orig_grid[:, :, np.newaxis].transpose(2, 0, 1)
    stacked_transform = np.tile(to_transform, (num_control_points, 1, 1))
    stacked_source_points = \
     source_points[:, :, np.newaxis].transpose(1, 0, 2)
    r_2 = np.sum((stacked_transform - stacked_source_points) ** 2, axis=1)

    # Take the product (r^2 * log(r^2)), being careful to avoid NaNs
    log_r_2 = np.log(r_2)
    log_r_2[np.isinf(log_r_2)] = 0.
    distances = r_2 * log_r_2
    print distances.shape

    # Add in the coefficients for the affine translation (1, x, and y,
    # corresponding to a_1, a_x, and a_y)
    upper_array = np.ones(shape=(1, orig_grid.shape[1]),
                       dtype=float)
    upper_array = np.concatenate([upper_array, orig_grid], axis=0)
    right_mat = np.concatenate([upper_array, distances], axis=0)

    # Convert to tensors
    # out_height = T.as_tensor_variable(out_height)
    # out_width = T.as_tensor_variable(out_width)
    # right_mat = T.as_tensor_variable(right_mat)

    upper_array_th=np.load('upper_array.npy');
    right_mat_th=np.load('right_array.npy');
    print np.max(np.abs(upper_array-upper_array_th));
    # print L[np.where(np.abs(L-L_th)>0.5)];
    print np.max(np.abs(right_mat-right_mat_th));

def stage_2():
    height=4;
    width=4;
    num_ctrl_pts=16;
    out_width=20;
    out_height=20;

    baseGrid = np.load('baseGrid.npy')
    batchGrid = np.load('batchGrid.npy')
    right_mat = np.load('right_mat.npy')
    L_inv = np.load('L_inv.npy')
    source_points = np.load('source_points.npy')

    batch_size=32;
    # dest_offsets=np.random.rand(batch_size,2*num_ctrl_pts);
    # np.save('dest_offsets.npy',dest_offsets);
    dest_offsets=np.load('dest_offsets.npy');
    t_x,t_y = _transform_thin_plate_spline(
        dest_offsets, right_mat, L_inv, source_points, out_height,
        out_width)
    
    print t_x.shape,t_y.shape;

    t_x_th=np.load('t_x.npy');
    t_y_th=np.load('t_y.npy');
    
    # t_x_th=np.sort(t_x_th);
    # t_x=np.sort(t_x);

    # t_y_th=np.sort(t_y_th);
    # t_y=np.sort(t_y);

    print np.max(np.abs(t_x-t_x_th));
    print np.max(np.abs(t_y-t_y_th));

    print ('hello');

def _transform_thin_plate_spline(
        dest_offsets, right_mat, L_inv, source_points, out_height,
        out_width):

    # num_batch, num_channels, height, width = input.shape
    num_control_points = source_points.shape[1]
    
    num_batch=dest_offsets.shape[0];

    # reshape destination offsets to be (num_batch, 2, num_control_points)
    # and add to source_points
    dest_points = source_points + np.reshape(
            dest_offsets, (num_batch, 2, num_control_points))
    print 'dest_points.shape', dest_points.shape
    # Solve as in ref [2]
    coefficients = np.dot(dest_points, L_inv[:, 3:].T)
    print 'coefficients.shape', coefficients.shape
    # Transform each point on the source grid (image_size x image_size)
    right_mat=np.expand_dims(right_mat,2);
    print 'right_mat.shape', right_mat.shape
    right_mat = np.tile(right_mat.transpose(2, 0, 1), (num_batch , 1, 1))
    print 'right_mat.shape', right_mat.shape

    transformed_points=np.zeros((num_batch,coefficients.shape[1],right_mat.shape[2]));
    for i in range(num_batch):
        transformed_points[i]=np.dot(coefficients[i],right_mat[i]);
    # transformed_points = np.tensordot(coefficients, right_mat,0)

    print 'transformed_points.shape', transformed_points.shape
    # Get out new points
    x_transformed = transformed_points[:, 0].flatten()
    y_transformed = transformed_points[:, 1].flatten()

    # dimshuffle input to  (bs, height, width, channels)
    # input_dim = input.transpose(0, 2, 3, 1)
    # # input_transformed = _interpolate(
    # #         input_dim, x_transformed, y_transformed,
    # #         out_height, out_width)

    # output = np.reshape(input_,
    #                    (num_batch, out_height, out_width, num_channels))
    # output = output.transpose(0, 3, 1, 2)  # dimshuffle to conv format
    return x_transformed,y_transformed


def _get_transformed_points_tps(new_points, source_points, coefficients,
                                num_points, batch_size):
    """
    Calculates the transformed points' value using the provided coefficients

    :param new_points: num_batch x 2 x num_to_transform tensor
    :param source_points: 2 x num_points array of source points
    :param coefficients: coefficients (should be shape (num_batch, 2,
        control_points + 3))
    :param num_points: the number of points

    :return: the x and y coordinates of each transformed point. Shape (
        num_batch, 2, num_to_transform)
    """

    # Calculate the U function for the new point and each source point as in
    # ref [2]
    # The U function is simply U(r) = r^2 * log(r^2), where r^2 is the
    # squared distance

    # Calculate the squared dist between the new point and the source points
    to_transform=np.expand_dims(new_points,3);
    to_transform = to_transform.transpose(0, 3, 1, 2)

    stacked_transform = np.tile(to_transform, (1, num_points, 1, 1))
    
    s_pts_t = np.expand_dims(np.expand_dims(source_points,2),3);

    r_2 = np.sum(((stacked_transform - s_pts_t.transpose(
            2, 1, 0, 3)) ** 2), axis=2)

    # Take the product (r^2 * log(r^2)), being careful to avoid NaNs
    log_r_2 = np.log(r_2)
    distances = np.where(np.isnan(log_r_2), r_2 * log_r_2, 0.)

    # Add in the coefficients for the affine translation (1, x, and y,
    # corresponding to a_1, a_x, and a_y)
    upper_array = np.concatenate([np.ones((batch_size, 1, new_points.shape[2]),
                                        dtype=theano.config.floatX),
                                 new_points], axis=1)
    right_mat = np.concatenate((upper_array, distances), axis=1)

    # Calculate the new value as the dot product
    new_value = np.tensordot(coefficients, right_mat,0)
    return new_value


def saveMatAsNpy(file_names,im_size=(32,32)):
    
    offsets=[];
    for file_name in file_names:
        print file_name
        x=scipy.io.loadmat(file_name);
        Xp=x['Xp'];
        Yp=x['Yp'];
        Xs=x['Xs'];
        Ys=x['Ys'];
        source=np.vstack((Xp.ravel(),Yp.ravel())).astype(float);
        dest=np.vstack((Xs.ravel(),Ys.ravel())).astype(float);
      
        source= (source/im_size[0]*2)-1;
        dest= (dest/im_size[0]*2)-1;
        offset= dest-source;
        # print offset
        print source;
        # print '___';
        # offset=offset[[1,0],:];
        # source=source[[1,0],:];
        # print offset
        # print source;
        offset=np.reshape(offset.T,(1,offset.shape[0]*offset.shape[1]));
        
        offsets.append(offset);
        # break;
    offsets=np.concatenate(offsets,axis=0);

    # print offsets.shape
    # print offsets[-1];
    np.save('dummy_offsets.npy',offsets);
    np.save('dummy_source.npy',source);

    
def main():
    out_dir='check_tps';
    dir_files='/Users/maheenrashid/Dropbox/Davis_docs/Research/horses/stnbhwd-master/demo/mnist_subset';
    file_neg_pos=os.path.join(dir_files,'neg_pos_grid.npy');
    grid_neg_pos=np.load(file_neg_pos)    
    # print grid_neg_pos;
    offsets=np.zeros(grid_neg_pos.shape);
    offsets[:,0]=2;
    offsets[:,1]=4;
    offsets_neg_pos=offsets/16;

    x = np.linspace(1, 32, 4);
    y = np.linspace(1, 32, 4);

    xv,yv=np.meshgrid(x,y);
    grid_pos=np.vstack((yv.ravel(),xv.ravel())).T;

    # print offsets;
    # print offsets_neg_pos;
    new_pos=grid_pos+offsets;
    dict_to_save={'Xp':grid_pos[:,0].T,'Yp':grid_pos[:,1].T,'Xs':new_pos[:,0].T,'Ys':new_pos[:,1].T};
    # print dict_to_save;
    # scipy.io.savemat(os.path.join(dir_files,'grid_pos.mat'),dict_to_save);

    source_pts=np.load('dummy_source.npy');
    print source_pts.shape;
    # dest_offsets=np.load('dummy_offsets.npy');

    offsets_neg_pos=offsets_neg_pos.ravel();
    offsets_neg_pos=np.tile(np.expand_dims(offsets_neg_pos,0),(10,1));
    print offsets_neg_pos.shape;
    print grid_neg_pos.shape;
    np.save('dummy_source_trans.npy',grid_neg_pos.T);
    np.save('dummy_dest_trans.npy',offsets_neg_pos);

    # print offsets_neg_pos


    # print dest_offsets.shape;

    # if not os.path.exists(out_dir):
    #     os.mkdir(out_dir);
    # offsets_int=((np.random.rand(10,2,16)*10)-5).astype(int);
    # print offsets_int[0];
    # offsets_fl=offsets_int/16.0;
    # print offsets_fl[0]


    


    return
    dir_files='/Users/maheenrashid/Dropbox/Davis_docs/Research/horses/stnbhwd-master/demo/mnist_subset';

    grids_th=np.load(os.path.join(dir_files,'dummy_grids.npy'))
    print (grids_th.shape);



    plt.ion();
    
    for i in range(1,11):
        file_name=os.path.join(dir_files,str(i)+'_outgrid.mat');
        mat=scipy.io.loadmat(file_name)
        Xw=mat['Xw'];
        Yw=mat['Yw'];
        print Xw.shape,Yw.shape
        X_g = grids_th[i-1,:,:,0];
        Y_g = grids_th[i-1,:,:,1];
        X_g = X_g.ravel();
        Y_g = Y_g.ravel();
        X_g=(X_g*16)+16;
        Y_g=(Y_g*16)+16;
        print (np.min(X_g),np.max(X_g),np.min(Xw),np.max(Xw));
        print (np.min(Y_g),np.max(Y_g),np.min(Yw),np.max(Yw));
        plt.figure();
        plt.subplot(1,2,1);
        plt.plot(X_g,Y_g);
        plt.subplot(1,2,2);
        plt.plot(Yw,Xw);
        plt.show();

    raw_input();




    # mat_files=[];
    # for i in range(1,11):
    #     mat_files.append(os.path.join(dir_files,str(i)+'.mat'));
    # # mat_files=[os.path.join(dir_files,file_curr) for file_curr in os.listdir(dir_files) if file_curr.endswith('.mat')];
    # saveMatAsNpy(mat_files);

    # stage_2();
    


if __name__=='__main__':
    main();
