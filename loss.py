import torch
import os
import torch.nn as nn
import numpy as np

from torchvision import models
from torchvision import transforms
# from places2_train import Places2Data, MEAN, STDDEV
from PIL import Image

LAMBDAS = {"valid": 1.0, "hole": 6.0, "tv": 0.1, "perceptual": 0.05, "style": 120.0}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def gram_matrix(input_tensor):
	"""
    Compute Gram matrix

    :param input_tensor: input tensor with shape
     (batch_size, nbr_channels, height, width)
    :return: Gram matrix of y
    """
	(b, ch, h, w) = input_tensor.size()
	features = input_tensor.view(b, ch, w * h)
	features_t = features.transpose(1, 2)

	# more efficient and formal way to avoid underflow for mixed precision training
	input = torch.zeros(b, ch, ch).type(features.type())
	gram = torch.baddbmm(input, features, features_t, beta=0, alpha=1./(ch * h * w), out=None)

	# naive way to avoid underflow for mixed precision training
	# features = features / (ch * h)
	# gram = features.bmm(features_t) / w

	# for fp32 training, it is also safe to use the following:
	# gram = features.bmm(features_t) / (ch * h * w)

	return gram


def perceptual_loss(h_comp, h_out, h_gt, l1):
	loss = 0.0

	for i in range(len(h_comp)):
		loss += l1(h_out[i], h_gt[i])
		loss += l1(h_comp[i], h_gt[i])

	return loss


def style_loss(h_comp, h_out, h_gt, l1):
	loss = 0.0

	for i in range(len(h_comp)):
		loss += l1(gram_matrix(h_out[i]), gram_matrix(h_gt[i]))
		loss += l1(gram_matrix(h_comp[i]), gram_matrix(h_gt[i]))

	return loss


# computes TV loss over entire composed image since gradient will not be passed backward to input
def total_variation_loss(image, l1):
    # shift one pixel and get loss1 difference (for both x and y direction)
    loss = l1(image[:, :, :, :-1] , image[:, :, :, 1:]) + l1(image[:, :, :-1, :] , image[:, :, 1:, :])
    return loss


class VGG16Extractor(nn.Module):
	def __init__(self):
		super().__init__()
		vgg16 = models.vgg16(weights = models.VGG16_Weights.DEFAULT).to(device)
		self.max_pooling1 = vgg16.features[:5]
		self.max_pooling2 = vgg16.features[5:10]
		self.max_pooling3 = vgg16.features[10:17]

		for i in range(1, 4):
			for param in getattr(self, 'max_pooling{:d}'.format(i)).parameters():
				param.requires_grad = False

	# feature extractor at each of the first three pooling layers
	def forward(self, image):
		results = [image]
		for i in range(1, 4):
			func = getattr(self, 'max_pooling{:d}'.format(i))
			results.append(func(results[-1]))
		return results[1:]


class CalculateLoss(nn.Module):
	def __init__(self):
		super().__init__()
		self.vgg_extract = VGG16Extractor()
		self.l1 = nn.L1Loss()

	def forward(self, input_x, mask, output, ground_truth):
		composed_output = (input_x * mask) + (output * (1 - mask))

		fs_composed_output = self.vgg_extract(composed_output)
		fs_output = self.vgg_extract(output)
		fs_ground_truth = self.vgg_extract(ground_truth)

		loss_dict = dict()

		loss_dict["valid"] = self.l1((1 - mask) * output, (1 - mask) * ground_truth) * LAMBDAS["valid"]
		loss_dict["hole"] = self.l1(mask * output, mask * ground_truth) * LAMBDAS["hole"]
		loss_dict["perceptual"] = perceptual_loss(fs_composed_output, fs_output, fs_ground_truth, self.l1) * LAMBDAS["perceptual"]
		loss_dict["style"] = style_loss(fs_composed_output, fs_output, fs_ground_truth, self.l1) * LAMBDAS["style"]
		loss_dict["tv"] = total_variation_loss(composed_output, self.l1) * LAMBDAS["tv"]

		# print(loss_dict)

		return sum(loss_dict.values())


# Unit Test
if __name__ == '__main__':
	from celeb import CelebDatasetFast
	from torch.utils.data import  DataLoader
	transform = transforms.Compose([transforms.PILToTensor(), transforms.Lambda(lambda x: x/255), transforms.Resize([178,178], antialias=True)])

	train_dataset = CelebDatasetFast(
		split='train', transform=transform)

	test_dataset = CelebDatasetFast(
		split='test', transform=transform)

	val_dataset = CelebDatasetFast(
		split='val', transform=transform)

	train_loader = DataLoader(train_dataset, 1, False)
	test_loader = DataLoader(test_dataset, 1, False)
	val_loader = DataLoader(val_dataset, 1, False)

	examples = iter(train_loader)
	samples = next(examples)
	mask,inp= samples[0]
	target = samples[1]
	output =torch.unsqueeze(torch.rand(3,178,178), 0) 

	hole_img = torch.mul(mask[0], target[0])
	out_img = torch.mul(1-mask[0], inp[0])

	image = transforms.ToPILImage()(hole_img)
	image.save("hole_img.png")
	image = transforms.ToPILImage()(out_img)
	image.save("inp_img.png")
	
	loss_func = CalculateLoss()
	loss_out = loss_func(inp.to(device),mask.to(device),target.to(device),target.to(device))
	print(loss_out)

    # for key, value in loss_out.items():
    #     print("KEY:{} | VALUE:{}".format(key, value))