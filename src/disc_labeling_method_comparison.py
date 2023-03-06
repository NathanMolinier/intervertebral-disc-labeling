import torch
from torchvision.utils import save_image
from test import retrieves_gt_coord, prediction_coordinates, check_skeleton, save_test_results
from pose_code.hourglass import hg
from pose_code.atthourglass import atthg
import numpy as np
import pickle
from torch.utils.data import DataLoader 
from train_utils import image_Dataset
import argparse
import cv2
import sys
from Metrics import mesure_err_disc, mesure_err_z, Faux_neg, Faux_pos
from sklearn.utils.extmath import cartesian
import os

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.utils.sys import run_proc

global CONTRAST
CONTRAST = {'t1': 'T1w',
            't2': 'T2w',
            't2s':'T2star'}

#---------------------------Test Sct Label Vertebrae--------------------------
def test_sct_label_vertebrae(args):
    '''
    Use sct_deepseg_sc and sct_label_vertebrae to find the vertebrae discs coordinates and append them
    to a txt file
    '''
    with open("prepared_data/discs_coords.txt","r") as f:  # Checking already processed subjects from coords.txt
        file_lines = f.readlines()
        processed_subjects_with_contrast = [line.split(' ')[0] + '_' + line.split(' ')[1] for line in file_lines[1:]]  # Remove first line
        
    #sct_coords = dict()
    datapath = os.path.abspath(args.sct_datapath)
    contrast = CONTRAST[args.modality]
    for dir_name in os.listdir(datapath):
        if dir_name.startswith('sub'):
            file_name = dir_name + '_' + contrast + '.nii.gz'
            file_path = os.path.join(datapath, dir_name, file_name)  # path to the original image
            seg_path = file_path.replace('.nii.gz', '_seg.nii.gz')  # path to the spinal cord segmentation
            if os.path.exists(seg_path):
                pass
            else:
                status, _ = run_proc(['sct_deepseg_sc',
                                        '-i', file_path, 
                                        '-c', args.modality,
                                        '-o', seg_path])
                if status != 0:
                    print('Fail segmentation')
                    discs_coords = 'Fail'
            
            disc_file_path = file_path.replace('.nii.gz', '_seg_labeled_discs.nii.gz')  # path to the file with disc labels
            if os.path.exists(disc_file_path):
                # retrieve all disc coords
                discs_coords = Image(disc_file_path).change_orientation("RPI").getNonZeroCoordinates(sorting='value')
            else:
                status, _ = run_proc(['sct_label_vertebrae',
                                            '-i', file_path,
                                            '-s', file_path.replace('.nii.gz', '_seg.nii.gz'),
                                            '-c', args.modality,
                                            '-ofolder', os.path.join(datapath, dir_name)], raise_exception=False)
                if status == 0:
                    discs_coords = Image(disc_file_path).change_orientation("RPI").getNonZeroCoordinates(sorting='value')
                else:
                    print('Exit value 1')
                    print('Fail sct_label_vertebrae')
                    discs_coords = 'Fail'

            subject_name = dir_name
            if (subject_name + '_' + contrast) not in processed_subjects_with_contrast:
                if discs_coords == 'Fail':  # SCT method error
                    lines = [subject_name + ' ' + contrast + ' ' + str(disc_num + 1) + ' ' + 'Fail' + ' ' + 'None' + ' ' + 'None' + '\n' for disc_num in range(11)] # To reorder the discs
                else:
                    lines = [subject_name + ' ' + contrast + ' ' + str(disc_num + 1) + ' ' + 'None' + ' ' + 'None' + ' ' + 'None' + '\n' for disc_num in range(11)] # To reorder the discs
                    last_referred_disc = 0
                    for coord in discs_coords:
                        coord_list = str(coord).split(',')
                        disc_num = int(float(coord_list[-1]))
                        coord_2d = '[' + str(coord_list[2]) + ',' + str(coord_list[1]) + ']'#  2D comparison of the models
                        if disc_num > 11:
                            print('More than 11 discs are visible')
                            print('Disc number', disc_num)
                            if disc_num == last_referred_disc + 1:  # Check if all the previous discs were also implemented
                                lines.append(subject_name + ' ' + contrast + ' ' + str(disc_num) + ' ' + coord_2d + ' ' + 'None' + ' ' + 'None' + '\n')
                                last_referred_disc = disc_num
                            else:
                                for i in range(disc_num - last_referred_disc - 1):
                                    lines.append(subject_name + ' ' + contrast + ' ' + str(last_referred_disc + 1 + i) + ' ' + 'None' + ' ' + 'None' + ' ' + 'None' + '\n')
                                lines.append(subject_name + ' ' + contrast + ' ' + str(disc_num) + ' ' + coord_2d + ' ' + 'None' + ' ' + 'None' + '\n')
                                last_referred_disc = disc_num
                        else:
                            lines[disc_num-1] = subject_name + ' ' + contrast + ' ' + str(disc_num) + ' ' + coord_2d + ' ' + 'None' + ' ' + 'None' + '\n'
                            last_referred_disc = disc_num
                with open("prepared_data/discs_coords.txt","a") as f:
                    f.writelines(lines)
            #sct_coords[subject_name] = discs_coords
    #return sct_coords

#---------------------------Test Hourglass Network----------------------------
def test_hourglass(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    contrast = CONTRAST[args.modality]
    
    print('load image')               
    with open(f'{args.hg_datapath}_test_{args.modality}', 'rb') as file_pi:
        full = pickle.load(file_pi)            
    
    full[0] = full[0][:, :, :, :, 0]
    coord_gt = retrieves_gt_coord(full[2])
    
    print('retrieving ground truth coordinates')
    global norm_mean_skeleton
    norm_mean_skeleton = np.load(f'./prepared_data/{args.modality}_Skelet.npy')
    #coord_gt = retrieves_gt_coord(ds) # [[182, 160, 139, 119, 100, 77, 57, 34, 9], [167, 167, 166, 164, 160, 155, 143, 129, 116]]
    
    # Initialize metrics
    metrics = dict()
    metrics['distance_l2'] = []
    metrics['zdis'] = []
    metrics['faux_pos'] = []
    metrics['faux_neg'] = []
    metrics['tot'] = []
    
    # Load network weights
    if args.att:
        model = atthg(num_stacks=args.stacks, num_blocks=args.blocks, num_classes=args.njoints)
        model = torch.nn.DataParallel(model).to(device)
        model.load_state_dict(torch.load(f'./weights/model_{args.modality}_att_stacks_{args.stacks}', map_location='cpu')['model_weights'])
    else:
        model = hg(num_stacks=args.stacks, num_blocks=args.blocks, num_classes=args.njoints)
        model = torch.nn.DataParallel(model).to(device)
        model.load_state_dict(torch.load(f'./weights/model_{args.modality}_stacks_{args.stacks}', map_location='cpu')['model_weights'])

    # Create Dataloader
    full_dataset_test = image_Dataset(image_paths=full[0],target_paths=full[1], gt_coords=full[2], subject_names=full[3], use_flip = False) 
    MRI_test_loader   = DataLoader(full_dataset_test, batch_size= 1, shuffle=False, num_workers=0)
    model.eval()
    
    # Load disc_coords txt file
    with open("prepared_data/discs_coords.txt","r") as f:  # Checking already processed subjects from coords.txt
        file_lines = f.readlines()
        split_lines = [line.split(' ') for line in file_lines]
    
    # Get the visualization results of the test set
    for i, (input, target, vis, gt_coord, subject_name) in enumerate(MRI_test_loader): # subject_name
        input, target = input.to(device), target.to(device, non_blocking=True)
        output = model(input) 
        output = output[-1]
        x      = full[0][i]
        
        prediction = extract_skeleton(input, output, target, Flag_save = True)
        prediction = np.sum(prediction[0], axis = 0)
        prediction = np.rot90(prediction,3)
        prediction = cv2.resize(prediction, (x.shape[0], x.shape[1]), interpolation=cv2.INTER_NEAREST)
        num_labels, labels_im, states, centers = cv2.connectedComponentsWithStats(np.uint8(np.where(prediction>0, 255, 0)))

        
        # Write the predicted and ground truth coordinates inside the discs_coords txt file
        pred = centers[1:] #0 for background
        pred = np.flip(pred[pred[:, 0].argsort()], axis=0)  # Sorting predictions according to first coordinate
        gt_coord = np.array(torch.tensor(gt_coord).tolist())
        gt_coord = np.transpose(np.array([gt_coord[:,2],gt_coord[:,1],gt_coord[:,-1]])) # Using same format as prediction + discs label
        subject_index = np.where((np.array(split_lines)[:,0] == subject_name[0]) & (np.array(split_lines)[:,1] == contrast))  
        start_index = subject_index[0][0]  # Getting the first line in the txt file
        
        pred, gt = best_disc_association(pred=pred, gt=gt_coord)
        for i in range(len(pred)):
            pred_coord = pred[i] if pred[i]!=0 else 'Fail'
            gt_coord = gt[i] if gt[i]!=0 else 'None'
            if pred_coord != 'Fail':
                split_lines[start_index + i][4] = '[' + str("{:.1f}".format(pred_coord[0])) + ',' + str("{:.1f}".format(pred_coord[1])) + ']'
            elif gt_coord == 'None':
                split_lines[start_index + i][4] = 'None'
            else:
                split_lines[start_index + i][4] = 'Fail'
            if gt_coord != 'None':
                split_lines[start_index + i][5] = '[' + str(gt_coord[0]) + ',' + str(gt_coord[1]) + ']' + '\n'
            else:
                split_lines[start_index + i][5] = 'None' + '\n'
                
        for num in range(len(split_lines)):
            file_lines[num] = ' '.join(split_lines[num])
            
        with open("prepared_data/discs_coords.txt","w") as f:
            f.writelines(file_lines)  
        
        # prediction_coordinates(prediction, gt_coord, metrics)
        

    # print('distance: l2_median = ' + str(np.median(metrics['distance_l2'])) + ', l2_std= ' + str(np.std(metrics['distance_l2'])))
    # print('distance: z_med= ' + str(np.mean(metrics['zdis'])) + ', z_std= ' + str(np.std(metrics['zdis'])))
    # print('faux neg (FN) per image ', metrics['faux_neg'])
    # print('total number of points ' + str(np.sum(metrics['tot'])))
    # print('number of faux neg (FN) ' + str(np.sum(metrics['faux_neg'])))
    # print('number of faux pos (FP) ' + str(np.sum(metrics['faux_pos'])))
    # print('False negative (FN) percentage ' + str(np.sum(metrics['faux_neg'])/ np.sum(metrics['tot'])*100))
    # print('False positive (FP) percentage ' + str(np.sum(metrics['faux_pos'])/ np.sum(metrics['tot'])*100))

##    
def extract_skeleton(inputs, outputs, target, Flag_save = False, target_th=0.5):
    global idtest
    idtest = 1
    outputs  = outputs.data.cpu().numpy()
    target  = target.data.cpu().numpy()
    inputs = inputs.data.cpu().numpy()
    skeleton_images = []
    for idx in range(outputs.shape[0]):    
        count_list = []
        Nch = 0
        center_list = {}
        while np.sum(np.sum(target[idx, Nch]))>0:
              Nch += 1       
        Final  = np.zeros((outputs.shape[0], Nch, outputs.shape[2], outputs.shape[3]))      
        for idy in range(Nch): 
            ych = outputs[idx, idy]
            ych = np.rot90(ych)
            ych = ych/np.max(np.max(ych))
            ych[np.where(ych<target_th)] = 0
            Final[idx, idy] = ych
            ych = np.where(ych>0, 1.0, 0)
            ych = np.uint8(ych)
            num_labels, labels_im, states, centers = cv2.connectedComponentsWithStats(ych)
            count_list.append(num_labels-1)
            center_list[str(idy)] = [t[::-1] for t in centers[1:]]
            
        ups = []
        for c in count_list:
            ups.append(range(c))
        combs = cartesian(ups)
        best_loss = np.Inf
        best_skeleton = []
        for comb in combs:
            cnd_skeleton = []
            for joint_idx, cnd_joint_idx in enumerate(comb):
                cnd_center = center_list[str(joint_idx)][cnd_joint_idx]
                cnd_skeleton.append(cnd_center)
            loss = check_skeleton(cnd_skeleton, norm_mean_skeleton)
            if best_loss > loss:
                best_loss = loss
                best_skeleton = cnd_skeleton
        Final2  = np.uint8(np.where(Final>0, 1, 0))
        cordimg = np.zeros(Final2.shape)
        hits = np.zeros_like(outputs[0])
        for i, jp, in enumerate(best_skeleton):
            jp = [int(t) for t in jp]
            hits[i, jp[0]-1:jp[0]+2, jp[1]-1:jp[1]+2] = [255, 255, 255]
            hits[i, :, :] = cv2.GaussianBlur(hits[i, :, :],(5,5),cv2.BORDER_DEFAULT)
            hits[i, :, :] = hits[i, :, :]/hits[i, :, :].max()*255
            cordimg[idx, i, jp[0], jp[1]] = 1
        
        for id_ in range(Final2.shape[1]):
            num_labels, labels_im = cv2.connectedComponents(Final2[idx, id_])
            for id_r in range(1, num_labels):
                if np.sum(np.sum((labels_im==id_r) * cordimg[idx, id_]) )>0:
                   labels_im = labels_im == id_r
                   continue
            Final2[idx, id_] = labels_im
        Final = Final * Final2           
                
        
        skeleton_images.append(hits)
        
    skeleton_images = np.array(skeleton_images)
    inputs = np.rot90(inputs, axes=(-2, -1))
    target = np.rot90(target, axes=(-2, -1))
    if Flag_save:
      save_test_results(inputs, skeleton_images, targets=target, name=idtest, target_th=0.5)
    idtest+=1
    return Final
##
def prediction_coordinates(final, coord_gt, metrics):
    num_labels, labels_im, states, centers = cv2.connectedComponentsWithStats(np.uint8(np.where(final>0, 255, 0)))
    #centers = peak_local_max(final, min_distance=5, threshold_rel=0.3)

    centers = centers[1:] #0 for background
    coordinates = []
    for x in centers:
        coordinates.append([x[0], x[1]])
    #print('calculating metrics on image')
    l2_dist = mesure_err_disc(coord_gt, coordinates)
    zdis = mesure_err_z(coord_gt, coordinates)
    fp, tot = Faux_pos(coord_gt, coordinates)
    fn = Faux_neg(coord_gt, coordinates)
    
    metrics['distance_l2'] += l2_dist  # Concatenation des listes
    metrics['zdis'] += zdis  # Concatenation des listes
    metrics['tot'].append(tot)
    metrics['faux_pos'].append(fp)
    metrics['faux_neg'].append(fn)
    
# looks for the closest points between real and predicted
def closest_node(node, nodes):
    nodes1 = np.asarray(nodes)
    dist_2 = np.sum((nodes1 - node) ** 2, axis=1)
    return np.argmin(dist_2), dist_2

# looks for the best association between ground truth and predicted discs
def best_disc_association(pred, gt):
    '''
    pred: numpy array of the coordinate of M discs
    gt: numpy array of the coordinate of N discs + num of the discs
    Note: M and N can be different
    
    return: Two lists (pred, gt) with the same length L corresponding to the biggest disc number in
    ground truth: L = gt[:,-1].max()
    '''
    M = pred.shape[0]
    N = gt.shape[0]
    L = gt[:,-1].max()
    pred_out, gt_out = [0]*L, [0]*L
    #if N >= M:
    dist_m = []
    for m in range(M):
        dist_m.append(np.sum((gt[:,:2] - pred[m]) ** 2, axis=1))
    dist_m = np.array(dist_m)
    ref_coord = []
    for n in range(N):
        disc_num = gt[n,-1]
        closer_to_node_n = np.argmin(dist_m[:,n])
        ref_coord.append([disc_num, closer_to_node_n, dist_m[closer_to_node_n,n]])
    ref_coord = np.array(ref_coord)
    new_ref_coord = []
    pred_coord_list = []
    for i in ref_coord:
        node_repetition = np.where((ref_coord[:,1]==i[1]))
        node = node_repetition[0][0]
        min_dist_node = ref_coord[node,2]
        if len(node_repetition[0]) > 1:
            for j in node_repetition[0][1:]:
                if ref_coord[j,2]<min_dist_node:
                    min_dist_node = ref_coord[j,2]
                    node = j
        if ref_coord[node,1] not in pred_coord_list:
            new_ref_coord.append(ref_coord[node])
            pred_coord_list.append(ref_coord[node,1])
    if len(pred_coord_list)<M:  # Every prediction point is not referenced
        for k in range(M):
            if k not in pred_coord_list:
                node, dist = closest_node(pred[k],gt[:,:2])
                closest_disc_num = gt[node,-1]
                if (closest_disc_num + 1) not in new_ref_coord[:][0]:
                    disc_num = closest_disc_num + 1
                    new_ref_coord.append([disc_num, k, dist])
                    
                elif (closest_disc_num - 1) not in new_ref_coord[:][0]:
                    disc_num = closest_disc_num - 1
                    new_ref_coord.append([disc_num, k, dist])
                    
                else:
                    print('Prediction disc error: discs might be misplaced, check disc:',closest_disc_num)
                
        
    if M > N: # TODO check this condition
        print('More discs detected by hourglass')
        print('nb_gt', N)
        print('nb_hourglass', M)
        print('PLZ CHECK THE SCRIPT')
        for j in range(M):
            if j not in new_ref_coord[:,1]: # Let's assume it's an extremity disc
                closest_gt, dist = closest_node(pred[j],gt[:,:2])
                if pred[j][0] < gt[closest_gt][0]:
                    disc_num = gt[closest_gt][-1] + 1
                    np.append(ref_coord,np.array([disc_num, j, dist]))
                else:
                    disc_num = gt[closest_gt][-1] - 1
                    if disc_num >= 1:
                        np.append(ref_coord,np.array([disc_num, j, dist]))
                    else:
                        print('Impossible disc prediction')
        for n in range(N):
            disc_num = int(gt[n,-1])
            gt_out[disc_num-1]=gt[n].tolist()
    else:
        
        for i in range(len(gt)):
            disc_num = gt[i][-1]
            gt_out[disc_num-1] = gt[i].tolist()
        
    for disc_num, closer_to_node_n, dist_m in new_ref_coord:
        disc_num = int(disc_num)
        closer_to_node_n = int(closer_to_node_n)      
        pred_out[disc_num-1]=pred[closer_to_node_n].tolist()
            
    return pred_out, gt_out
            
                    
        
        
    
def compare_methods(args):
    contrast = CONTRAST[args.modality]
    txt_file_path = args.comp_txt_file
    
    # Load disc_coords txt file
    with open(txt_file_path,"r") as f:  # Checking already processed subjects from coords.txt
        file_lines = f.readlines()
        split_lines = [line.split(' ') for line in file_lines]
    
    return

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Verterbal disc labeling using pose estimation')

    ## Parameters
    parser.add_argument('--hg_datapath', type=str,
                        help='Hourglass dataset path')                               
    parser.add_argument('--sct_datapath', type=str,
                        help='SCT dataset path')                               
    parser.add_argument('-c', '--modality', type=str, metavar='N', required=True,
                        help='Data modality')
    parser.add_argument('-txt', '--comp_txt_file', default=None, type=str, metavar='N',
                        help='Data modality')                                                                                                

    parser.add_argument('--njoints', default=11, type=int,
                        help='Number of joints')
    parser.add_argument('--resume', default= False, type=bool,
                        help=' Resume the training from the last checkpoint') 
    parser.add_argument('--att', default= True, type=bool,
                        help=' Use attention mechanism') 
    parser.add_argument('--stacks', default=2, type=int, metavar='N',
                        help='Number of hourglasses to stack')
    parser.add_argument('-b', '--blocks', default=1, type=int, metavar='N',
                        help='Number of residual modules at each location in the hourglass')
    
    if parser.parse_args().comp_txt_file != None:
        compare_methods(parser.parse_args())
    else:
        if not os.path.exists('prepared_data/discs_coords.txt'):
            with open("prepared_data/discs_coords.txt","w") as f:
                f.write("subject_name contrast num_disc sct_discs_coords hourglass_coords gt_coords\n")

        test_sct_label_vertebrae(parser.parse_args())
        test_hourglass(parser.parse_args())