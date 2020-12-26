import sys
sys.path.append('/home/deliware/Desktop/jetson_inference/jetson-inference/python/examples')

import jetson.inference
import jetson.utils
from segnet_utils import segmentationBuffers


class Args:
    def __init__(self):
        self.stats = True
        self.visualize = "overlay,mask"


class Segmentor:

    def __init__(self,
                 input_image_shape,
                 network_name='fcn-resnet18-cityscapes'):
        self.net = jetson.inference.segNet(network_name)
        args = Args()
        self.buffers = segmentationBuffers(self.net, args)
        self.buffers.Alloc(input_image_shape, 'rgb8')

    def do_segmentation(self, rgb_uint8_img, output_img_name):
        cuda_img = jetson.utils.cudaFromNumpy(rgb_uint8_img, isBGR=False)
        self.net.Process(cuda_img, ignore_class='void')
        jetson.utils.saveImage(f'{output_img_name}.jpg', self.buffers.mask)