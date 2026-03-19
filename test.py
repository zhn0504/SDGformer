import os
import argparse

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from skimage import img_as_ubyte

import utils
from data_RGB import get_test_data
from layers import window_partitionx, window_reversex
from models.model import SDGformer


parser = argparse.ArgumentParser(description='Image Deraining using SDGformer')

# Input directory that contains benchmark sub-folders (e.g., Test2800/Test1200/Rain100L/...)
parser.add_argument('--input_dir', default='./Datasets/Synthetic_Rain_Datasets/test/', type=str,
                    help='Directory of testing images')
parser.add_argument('--result_dir', default='/root/autodl-fs/results', type=str,
                    help='Directory for saving restored images')
parser.add_argument('--weights', default='/root/autodl-fs/checkpoints/Deraining/models/SDGformer/SDGformer_best.pth',
                    type=str, help='Path to model weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--model', default='SDGformer', type=str, help='Sub-directory name for results')
args = parser.parse_args()


def get_main_output(restored):
    if isinstance(restored, (list, tuple)):
        return restored[-1]
    return restored


def main():
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    model_restoration = SDGformer()
    utils.load_checkpoint(model_restoration, args.weights)
    print('===> Testing using weights:', args.weights)

    model_restoration.cuda()
    model_restoration = nn.DataParallel(model_restoration)
    model_restoration.eval()

    win = 128  # window size
    datasets = ['Test2800', 'Test1200', 'Rain100L', 'Rain100H', 'Test100']

    for dataset in datasets:
        rgb_dir_test = os.path.join(args.input_dir, dataset, 'input')
        test_dataset = get_test_data(rgb_dir_test, img_options={})
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            pin_memory=True,
        )

        result_dir = os.path.join(args.result_dir, args.model, dataset)
        utils.mkdir(result_dir)

        with torch.no_grad():
            for _, data_test in enumerate(tqdm(test_loader), 0):
                input_ = data_test[0].cuda(non_blocking=True)
                filenames = data_test[1]
                _, _, Hx, Wx = input_.shape

                input_re, batch_list = window_partitionx(input_, win)
                restored = get_main_output(model_restoration(input_re))
                restored = window_reversex(restored, win, Hx, Wx, batch_list)

                restored = torch.clamp(restored, 0, 1)
                restored = restored.permute(0, 2, 3, 1).cpu().numpy()

                for batch in range(len(restored)):
                    restored_img = img_as_ubyte(restored[batch])
                    utils.save_img(os.path.join(result_dir, filenames[batch] + '.png'), restored_img)


if __name__ == '__main__':
    main()
