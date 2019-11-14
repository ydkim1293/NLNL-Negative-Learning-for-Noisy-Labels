import args
import os
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms
import random
import pickle

opt = args.args()

try:
    os.makedirs('noise/%s'%(opt.noise_type))
except OSError:
    pass
###################################################################################################
if   opt.dataset == 'cifar10_wo_val' : num_classes = 10
else: print('There exists no data')

trainset = dset.ImageFolder(root='{}/{}/train'.format(opt.dataroot, opt.dataset), transform=transforms.ToTensor())
clean_labels = np.array(trainset.imgs)[:,1]

for n in range(10):

	trainset =dset.ImageFolder(root='{}/{}/train'.format(opt.dataroot, opt.dataset), transform=transforms.ToTensor())

	noisy_idx = []
	for c in range(num_classes):
		noisy_idx.extend(random.sample(list(np.where(clean_labels.astype(int) == c)[0]), int(len(trainset.imgs)*(n*0.1/num_classes))))

	trainset.imgs_temp = np.empty_like(trainset.imgs) # to change tuple to list

	trainset.imgs_temp = [list(trainset.imgs[i]) for i in range(len(trainset.imgs))]
	trainset.imgs = trainset.imgs_temp

	for i in noisy_idx:

		if 'symm_exc' in opt.noise_type:
			trainset.imgs[i][1] = random.sample(list(range(0,trainset.imgs[i][1]))+list(range(trainset.imgs[i][1]+1,num_classes)),1)[0]

		elif 'asymm' in opt.noise_type:
  			if opt.dataset == 'cifar10_wo_val':
  				if   trainset.imgs[i][1] == 9: trainset.imgs[i][1] = 1
  				elif trainset.imgs[i][1] == 2: trainset.imgs[i][1] = 0
  				elif trainset.imgs[i][1] == 3: trainset.imgs[i][1] = 5
  				elif trainset.imgs[i][1] == 5: trainset.imgs[i][1] = 3
  				elif trainset.imgs[i][1] == 4: trainset.imgs[i][1] = 7  

	noisy_labels = np.array(trainset.imgs)[:,1]

	print(float(np.sum(clean_labels != noisy_labels)) / len(clean_labels))
	with open('noise/%s/train_labels_n%02d_%s'%(opt.noise_type, n*10, opt.dataset), 'wb') as fp:
		pickle.dump(noisy_labels, fp)

	for k in range(num_classes): # Checking class-wise noise ratio
		clean_int = clean_labels.astype(int)
		noisy_int = noisy_labels.astype(int)
		print(k, len(clean_int[clean_int==k]), float(np.sum(clean_int[clean_int==k] != noisy_int[clean_int==k])) / len(clean_int[clean_int==k]))

	leng = int(len(trainset)/num_classes) 
	for k in range(num_classes): # Checking class-wise noise
		print(noisy_labels[leng*k:leng*k+15])