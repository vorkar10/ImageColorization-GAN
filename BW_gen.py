import os
import cv2

class_list = os.listdir('flowers')
print(class_list)

import matplotlib.pyplot as plt

data = {}

for folder in class_list:
	if folder not in data:
		data[folder]=0
	for i, img in enumerate(os.listdir(os.path.join('flowers',folder))):
		data[folder]+=1
	for i, img in enumerate(os.listdir(os.path.join('flowers',folder))):
		image = cv2.imread(os.path.join('flowers',folder,img))
		image = cv2.resize(image, (256,256), interpolation = cv2.INTER_AREA)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		backtorgb = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
		im_h = cv2.hconcat([backtorgb,image])
		# class_img = img.split('_')[0]
		if i%10==0:
			if not os.path.exists('test'):
				os.mkdir('test')
			if not os.path.exists(os.path.join('test',folder)):
				os.mkdir(os.path.join('test',folder))
			cv2.imwrite(os.path.join('test',folder,img), im_h)
		else:
			if not os.path.exists('train'):
				os.mkdir('train')
			if not os.path.exists(os.path.join('train',folder)):
				os.mkdir(os.path.join('train',folder))
			cv2.imwrite(os.path.join('train',folder,img), im_h)


names = list(data.keys())
values = list(data.values())

plt.bar(range(len(data)), values, tick_label=names)
plt.show()


