import argparse
import numpy as np

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
    for subject in processed_subjects:
        discs_list = np.extract(split_lines[:,0] == subject,split_lines[:,2])
        sct_coords_list = np.extract(split_lines[:,0] == subject,split_lines[:,3])
        hg_coords_list = np.extract(split_lines[:,0] == subject,split_lines[:,4])
        gt_coords_list = np.extract(split_lines[:,0] == subject,split_lines[:,5])
            

    return

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Compute metrics on sct and hourglass disc estimation')
    
    parser.add_argument('-txt', '--input_txt_file', default='prepared_data/discs_coords.txt', type=str, metavar='N',
                        help='Input txt file with the methods coordinates') 
    parser.add_argument('-c', '--modality', type=str, metavar='N', required=True,
                        help='Data modality')
    
    compare_methods(parser.parse_args())