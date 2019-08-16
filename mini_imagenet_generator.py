##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## NUS School of Computing
## Email: yaoyao.liu@nus.edu.sg
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import argparse
import os
import numpy as np
import csv
import glob
import cv2
from shutil import copyfile
from tqdm import tqdm

# argument parser
parser = argparse.ArgumentParser(description='')
parser.add_argument('--tar_dir',  type=str)
parser.add_argument('--imagenet_dir',  type=str)
parser.add_argument('--image_resize',  type=int,  default=84)

args = parser.parse_args()

class MiniImageNetGenerator(object):
    def __init__(self, input_args):
        self.input_args = input_args
        if self.input_args.tar_dir is not None:
            print('Untarring ILSVRC2012 package')
            self.imagenet_dir = './imagenet'
            if not os.path.exists(self.imagenet_dir):
                os.mkdir(self.imagenet_dir)
            os.system('tar xvf ' + str(self.input_args.tar_dir) + ' -C ' + self.imagenet_dir)
        elif self.input_args.imagenet_dir is not None:
            self.imagenet_dir = self.input_args.imagenet_dir
        else:
            print('You need to specify the ILSVRC2012 source file path')
        self.mini_dir = './mini_imagenet'
        if not os.path.exists(self.mini_dir):
            os.mkdir(self.mini_dir)
        self.image_resize = self.input_args.image_resize
        
    def untar_mini(self):
        self.mini_keys = ['n02110341', 'n01930112', 'n04509417', 'n04067472', 'n04515003', 'n02120079', 'n03924679', 'n02687172', 'n03075370', 'n07747607', 'n09246464', 'n02457408', 'n04418357', 'n03535780', 'n04435653', 'n03207743', 'n04251144', 'n03062245', 'n02174001', 'n07613480', 'n03998194', 'n02074367', 'n04146614', 'n04243546', 'n03854065', 'n03838899', 'n02871525', 'n03544143', 'n02108089', 'n13133613', 'n03676483', 'n03337140', 'n03272010', 'n01770081', 'n09256479', 'n02091244', 'n02116738', 'n04275548', 'n03773504', 'n02606052', 'n03146219', 'n04149813', 'n07697537', 'n02823428', 'n02089867', 'n03017168', 'n01704323', 'n01532829', 'n03047690', 'n03775546', 'n01843383', 'n02971356', 'n13054560', 'n02108551', 'n02101006', 'n03417042', 'n04612504', 'n01558993', 'n04522168', 'n02795169', 'n06794110', 'n01855672', 'n04258138', 'n02110063', 'n07584110', 'n02091831', 'n03584254', 'n03888605', 'n02113712', 'n03980874', 'n02219486', 'n02138441', 'n02165456', 'n02108915', 'n03770439', 'n01981276', 'n03220513', 'n02099601', 'n02747177', 'n01749939', 'n03476684', 'n02105505', 'n02950826', 'n04389033', 'n03347037', 'n02966193', 'n03127925', 'n03400231', 'n04296562', 'n03527444', 'n04443257', 'n02443484', 'n02114548', 'n04604644', 'n01910747', 'n04596742', 'n02111277', 'n03908618', 'n02129165', 'n02981792']
        
        for idx, keys in enumerate(self.mini_keys):
            print('Untarring ' + keys)
            os.system('tar xvf ' + self.imagenet_dir + '/' + keys + '.tar -C ' + self.mini_dir)
        print('All the tar files are untarred')

    def process_original_files(self):
        self.processed_img_dir = './processed_images'
        split_lists = ['train', 'val', 'test']
        csv_files = ['./csv_files/train.csv','./csv_files/val.csv', './csv_files/test.csv']

        if not os.path.exists(self.processed_img_dir):
            os.makedirs(self.processed_img_dir)

        for this_split in split_lists:
            filename = './csv_files/' + this_split + '.csv'
            this_split_dir = self.processed_img_dir + '/' + this_split
            if not os.path.exists(this_split_dir):
                os.makedirs(this_split_dir)
            with open(filename) as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=',')
                next(csv_reader, None)
                images = {}
                print('Reading IDs....')

                for row in tqdm(csv_reader):
                    if row[1] in images.keys():
                        images[row[1]].append(row[0])
                    else:
                        images[row[1]] = [row[0]]

                print('Writing photos....')
                for cls in tqdm(images.keys()):
                    this_cls_dir = this_split_dir + '/' + cls        
                    if not os.path.exists(this_cls_dir):
                        os.makedirs(this_cls_dir)

                    lst_files = []
                    for file in glob.glob(self.mini_dir + "/*"+cls+"*"):
                        lst_files.append(file)

                    lst_index = [int(i[i.rfind('_')+1:i.rfind('.')]) for i in lst_files]
                    index_sorted = sorted(range(len(lst_index)), key=lst_index.__getitem__)

                    index_selected = [int(i[i.index('.') - 4:i.index('.')]) for i in images[cls]]
                    selected_images = np.array(index_sorted)[np.array(index_selected) - 1]
                    for i in np.arange(len(selected_images)):
                        if self.image_resize==0:
                            copyfile(lst_files[selected_images[i]],os.path.join(this_cls_dir, images[cls][i]))
                        else:
                            im = cv2.imread(lst_files[selected_images[i]])
                            im_resized = cv2.resize(im, (self.image_resize, self.image_resize), interpolation=cv2.INTER_AREA)
                            cv2.imwrite(os.path.join(this_cls_dir, images[cls][i]),im_resized)

if __name__ == "__main__":
    dataset_generator = MiniImageNetGenerator(args)
    dataset_generator.untar_mini()
    dataset_generator.process_original_files()
