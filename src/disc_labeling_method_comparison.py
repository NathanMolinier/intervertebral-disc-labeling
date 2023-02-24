import torch
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

sys.path.append(r'/home/nathanmolinier/data_nvme/code/spinalcordtoolbox')
from spinalcordtoolbox.scripts.sct_label_vertebrae import main as sct_label_vertebrae
from spinalcordtoolbox.scripts.sct_deepseg_sc import main as sct_deepseg_sc
from spinalcordtoolbox.image import Image

#---------------------------Test Sct Label Vertebrae--------------------------
def test_sct_label_vertebrae(args):
    '''
    Use sct_deepseg_sc and sct_label_vertebrae to find the vertebrae discs coordinates and append them
    to a txt file
    '''
    #coords_txt = open('/home/nathanmolinier/data_nvme/code/intervertebral-disc-labeling/prepared_data/coords.txt','a+')
    #coords_txt.write('subject_name sct_discs_coords hourglass_coords gt_coords')
    sct_coords = dict()
    datapath = os.path.abspath(args.sct_datapath)
    if args.modality == 't1':
        contrast = 'T1w'
    elif args.modality == 't2':
        contrast = 'T2w'
    elif args.modality == 't2s':
        contrast = 'T2star'
    for dir_name in os.listdir(datapath):
        if dir_name.startswith('sub'):
            file_name = dir_name + '_' + contrast + '.nii.gz'
            file_path = os.path.join(datapath, dir_name, file_name)  # path to the original image
            seg_path = file_path.replace('.nii.gz', '_seg.nii.gz')  # path to the spinal cord segmentation
            if os.path.exists(seg_path):
                pass
            else:
                sct_deepseg_sc(argv=['-i', file_path, 
                                    '-c', args.modality,
                                    '-o', seg_path])
            
            disc_file_path = file_path.replace('.nii.gz', '_seg_labeled_discs.nii.gz')  # path to the file with disc labels
            if os.path.exists(disc_file_path):
                pass
            else:
                sct_label_vertebrae(argv=['-i', file_path,
                                        '-s', file_path.replace('.nii.gz', '_seg.nii.gz'),
                                        '-c', args.modality,
                                        '-ofolder', os.path.join(datapath, dir_name)])
            # retrieve all disc coords
            disc_coords = Image(disc_file_path).change_orientation("RPI").getNonZeroCoordinates()
            subject_name = file_name.replace('.nii.gz', '')
            sct_coords[subject_name] = disc_coords
    return sct_coords

#---------------------------Test Hourglass Network----------------------------
def test_hourglass(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print('load image')
    # Put image into an array
    with open(f'{args.hg_datapath}_{args.modality}_ds',   'rb') as file_pi:       
         ds = pickle.load(file_pi)
    with open(f'{args.hg_datapath}_{args.modality}_full', 'rb') as file_pi:
         full = pickle.load(file_pi)            
               
    full[0] = full[0][:, :, :, :, 0]
    
    print('retrieving ground truth coordinates')
    global norm_mean_skeleton
    norm_mean_skeleton = np.load(f'./prepared_data/{args.modality}_Skelet.npy')
    coord_gt = retrieves_gt_coord(ds)
    
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
    full_dataset_test = image_Dataset(image_paths=full[0],target_paths=full[1], subject_names=full[2], use_flip = False)
    MRI_test_loader   = DataLoader(full_dataset_test, batch_size= 1, shuffle=False, num_workers=0)
    model.eval()
    
    # Get the visualization results of the test set
    for i, (input, target, vis, subject_name) in enumerate(MRI_test_loader):
        input, target = input.to(device), target.to(device, non_blocking=True)
        output = model(input) 
        output = output[-1]
        x      = full[0][i]
        
        prediction = extract_skeleton(input, output, target, Flag_save = True)
        prediction = np.sum(prediction[0], axis = 0)
        prediction = np.rot90(prediction,3)
        prediction = cv2.resize(prediction, (x.shape[0], x.shape[1]), interpolation=cv2.INTER_NEAREST)
        prediction_coordinates(prediction, coord_gt[i], metrics)
        print(subject_name, prediction, coord_gt[i])

    print('distance: l2_median = ' + str(np.median(metrics['distance_l2'])) + ', l2_std= ' + str(np.std(metrics['distance_l2'])))
    print('distance: z_med= ' + str(np.mean(metrics['zdis'])) + ', z_std= ' + str(np.std(metrics['zdis'])))
    print('faux neg (FN) per image ', metrics['faux_neg'])
    print('total number of points ' + str(np.sum(metrics['tot'])))
    print('number of faux neg (FN) ' + str(np.sum(metrics['faux_neg'])))
    print('number of faux pos (FP) ' + str(np.sum(metrics['faux_pos'])))
    print('False negative (FN) percentage ' + str(np.sum(metrics['faux_neg'])/ np.sum(metrics['tot'])*100))
    print('False positive (FP) percentage ' + str(np.sum(metrics['faux_pos'])/ np.sum(metrics['tot'])*100))

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
    
def compare_methods():
    return

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Verterbal disc labeling using pose estimation')

    ## Parameters
    parser.add_argument('--hg_datapath', type=str, required=True,
                        help='Hourglass dataset path')                               
    parser.add_argument('--sct_datapath', type=str, required=True,
                        help='SCT dataset path')                               
    parser.add_argument('-c', '--modality', type=str, metavar='N', required=True,
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

    # test_sct_label_vertebrae(parser.parse_args())
    test_hourglass(parser.parse_args())