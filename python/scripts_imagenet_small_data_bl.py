import util;
import os;
import numpy as np;

def script_make_small_data_paths():
	dir_old='/home/SSD3/maheen-data/horse_project/neighbor_data/small_datasets';
	file_pre_post=['matches_5_','_horse_minloss.txt'];
	range_data=range(500,3500,500);
	range_data.append(3531);
	files_old=[os.path.join(dir_old,file_pre_post[0]+str(num_data)+file_pre_post[1]) for num_data in range_data];
	# for file_curr in files_old:
	# 	print file_curr,os.path.exists(file_curr);

	out_dir='../data_227';
	file_pre='small_train_minloss_';
	old_path_data='/home/SSD3/maheen-data/horse_project/data_check';
	new_path_data='../data_227';
	for num_data,file_curr in zip(range_data,files_old):
		lines=util.readLinesFromFile(file_curr);
		lines=[line_curr.replace(old_path_data,new_path_data) for line_curr in lines];
		# print lines[0];
		# print os.path.exists(lines[0].split(' ')[0]);
		out_file=os.path.join(out_dir,file_pre+str(num_data)+'.txt');
		print num_data;
		# print file_curr;
		# print out_file;
		util.writeFile(out_file,lines);
		print 'done';


def script_train_alexnet():
	# script_make_small_data_paths();
	out_file_script='../scripts/imagenet_last_scratch_small_data';
	num_scripts=2;
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
		params=['th',torch_file];
		params.extend(['-outDir',dir_curr]);
		params.extend(['-data_path',train_file]);
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

def script_test_alexnet():
	out_file_script='../scripts/imagenet_last_scratch_small_data_test';
	num_scripts=2;
	torch_file='test.th';
	out_dir_meta='/home/SSD3/maheen-data/horse_project/cvpr_rebuttal/imagenet_last_scratch_small_data';
	util.mkdir(out_dir_meta);
	train_data_dir='../data_227';
	train_file_pre='small_train_minloss_'
	range_data=range(500,3500,500);
	range_data.append(3531);

	# train_file_pre=
	# os.path.join(train_data_dir,train_file_pre);
	test_file=os.path.join(train_data_dir,'test_minloss_horse_old.txt');
	# cmd:option('-val_data_path','../data_227/test_minloss_horse_old.txt','validation data file path');
	# cmd:option('-model_path','/home/SSD3/maheen-data/horse_project/cvpr_rebuttal/alexnet_fc_scratch_ft/final/model_all_final.dat');
	# cmd:option('-out_dir_images','/home/SSD3/maheen-data/horse_project/cvpr_rebuttal/alexnet_fc_scratch_ft/test_images');


	commands=[];
	for num_data in range_data:
		dir_curr=os.path.join(out_dir_meta,train_file_pre+str(num_data),'test_images');
		model_file=os.path.join(out_dir_meta,train_file_pre+str(num_data),'final','model_all_final.dat');
		params=['th',torch_file];
		params.extend(['-model_path',model_file]);
		params.extend(['-out_dir_images',dir_curr]);
		params.extend(['-val_data_path',test_file]);
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

def main():
	out_file_script='../scripts/imagenet_last_scratch_small_data_viz';
	num_scripts=2;
	torch_file='visualize_results.py';
	out_dir_meta='/home/SSD3/maheen-data/horse_project/cvpr_rebuttal/imagenet_last_scratch_small_data';
	util.mkdir(out_dir_meta);
	train_data_dir='../data_227';
	train_file_pre='small_train_minloss_'
	range_data=range(500,3500,500);
	range_data.append(3531);

	test_file=os.path.join(train_data_dir,'test_minloss_horse_old.txt');
	batchSize=100;
	iterations=2;
	
	commands=[];
	for num_data in range_data:
		dir_curr=os.path.join(out_dir_meta,train_file_pre+str(num_data),'test_images');
		params=['python',torch_file];
		params.extend(['--test_dir',dir_curr]);
		params.extend(['--test_file',test_file]);
		params.extend(['--batchSize',str(batchSize)]);
		params.extend(['--iterations',str(iterations)]);
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
	
	



if __name__=='__main__':
	main();