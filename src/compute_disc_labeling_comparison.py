import argparse
import numpy as np
import matplotlib.pyplot as plt

from Metrics import compute_L2_error
from extract_discs_coords import CONTRAST

def compare_methods(args):
    contrast = CONTRAST[args.modality]
    txt_file_path = args.input_txt_file
    
    # Load disc_coords txt file
    with open(txt_file_path,"r") as f:  # Checking already processed subjects from txt file
        file_lines = f.readlines()
        split_lines = np.array([line.split(' ') for line in file_lines])
    
    # Extract subjects processed by sct and hourglass method
    processed_subjects = []
    for line in split_lines[1:]:
        if (line[0] not in processed_subjects) and (line[1]==contrast) and (line[5]!='None\n'):
            processed_subjects.append(line[0])
    
    # Compute metrics on coordinates
    '''
    metrics['distance_l2'] += l2_dist  # Concatenation des listes
    metrics['zdis'] += zdis  # Concatenation des listes
    metrics['tot'].append(tot)
    metrics['faux_pos'].append(fp)
    metrics['faux_neg']
    '''
    methods_results = {} 
    l2_mean_hg = []
    l2_mean_sct = []
    fail_hg = []
    fail_sct = []
    for subject in processed_subjects:
        # Convert str coords to numpy array
        discs_list = np.extract(split_lines[:,0] == subject,split_lines[:,2]).astype(int)
        sct_coords_list = str2array(np.extract(split_lines[:,0] == subject,split_lines[:,3]))
        hg_coords_list = str2array(np.extract(split_lines[:,0] == subject,split_lines[:,4]))
        gt_coords_list = str2array(np.extract(split_lines[:,0] == subject,split_lines[:,5]))
        
        # Add subject to result dict
        methods_results[subject] = {}
        
        # Compute L2 error
        L2_hourglass = compute_L2_error(gt=gt_coords_list, pred=hg_coords_list)
        L2_sct = compute_L2_error(gt=gt_coords_list, pred=sct_coords_list)
        
        fail_detection_hg, fail_detection_sct = np.count_nonzero(L2_hourglass == -2), np.count_nonzero(L2_sct == -2)
        no_gt_coords = np.count_nonzero(L2_sct == -1)
        
        idx_neg_hg = np.in1d(L2_hourglass, np.array([-1, -2])) # Get position of all negative value in L2_hourglass
        idx_neg_sct = np.in1d(L2_sct, np.array([-1, -2])) # Get position of all negative value in L2_sct
        L2_hourglass_pos = L2_hourglass[~idx_neg_hg] # Get only positive value 
        L2_sct_pos = L2_sct[~idx_neg_sct] # Get only positive value 
        
        L2_hourglass_mean = np.mean(L2_hourglass_pos)
        L2_sct_mean = np.mean(L2_sct_pos)
        L2_hourglass_std = np.std(L2_hourglass_pos)
        L2_sct_std = np.std(L2_sct_pos)

        # Add computed metrics to subject
        methods_results[subject]['l2_mean_hg'] = L2_hourglass_mean
        methods_results[subject]['l2_mean_sct'] = L2_sct_mean
        methods_results[subject]['l2_std_hg'] = L2_hourglass_std
        methods_results[subject]['l2_std_sct'] = L2_sct_std
        
        methods_results[subject]['fail_det_hg'] = fail_detection_hg
        methods_results[subject]['fail_det_sct'] = fail_detection_sct
        methods_results[subject]['total_gt_discs'] = gt_coords_list.shape[0]-no_gt_coords
        
        l2_mean_hg.append(L2_hourglass_mean)
        l2_mean_sct.append(L2_sct_mean)
        fail_hg.append(fail_detection_hg)
        fail_sct.append(fail_detection_sct)
    
    # set width of bar
    barWidth = 0.25
    fig = plt.subplots(figsize =(30, 4))
    
    # Set position of bar on X axis
    br1 = np.arange(len(processed_subjects))
    br2 = [x + barWidth for x in br1]
    
    # Make the plot        
    # plt.bar(br1, l2_mean_sct, color='r', width = barWidth, edgecolor ='grey', label ='SCT_label_vertebrae')
    # plt.bar(br2, l2_mean_hg, color='b', width = barWidth, edgecolor ='grey', label ='Hourglass_network')
    plt.bar(br1, fail_sct, color='r', width = barWidth, edgecolor ='grey', label ='SCT_label_vertebrae')
    plt.bar(br2, fail_hg, color='b', width = barWidth, edgecolor ='grey', label ='Hourglass_network')
     
    
    # Create axis and adding Xticks
    # plt.xlabel('Subjects', fontweight ='bold', fontsize = 15)
    # plt.ylabel('L2_error (pixels)', fontweight ='bold', fontsize = 15)
    plt.ylabel('Fail detections', fontweight ='bold', fontsize = 15)
    plt.xticks([r + barWidth/2 for r in range(len(processed_subjects))], processed_subjects)
    
    # Show plot
    plt.legend()
    plt.show()
    plt.savefig('prepared_data/fail_detection.png')        
                

    return

def str2array(coords):
    '''
    coords: numpy array of str coords
    output_coords: numpy array of coordinates
    '''
    output_coords = []
    for coord in coords:
        if coord not in ['Fail', 'None', 'None\n']:
            coord_split = coord.split(',')
            output_coords.append([float(coord_split[0].split('[')[1]),float(coord_split[1].split(']')[0])])
        else:
            output_coords.append([-1,-1])
    return np.array(output_coords)
    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Compute metrics on sct and hourglass disc estimation')
    
    parser.add_argument('-txt', '--input_txt_file', default='prepared_data/discs_coords.txt', type=str, metavar='N',
                        help='Input txt file with the methods coordinates') 
    parser.add_argument('-c', '--modality', type=str, metavar='N', required=True,
                        help='Data modality')
    
    compare_methods(parser.parse_args())