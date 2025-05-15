import os
from PIL import Image

# for ERP image resize
root_path = '/home1/mpc/Dataset/OIQA/distorted_images/'
index = 1
img_path = root_path + 'img' + str(index) + '.jpg'
img = Image.open(img_path)
img = img.resize((1024, 512))
save_root = './resized_img/'
if not os.path.exists(save_root):
    os.mkdir(save_root)
save_path = save_root + str(index).rjust(3, '0') + '.png'
img.save(save_path)

# for viewport images resize
root_path = '/home1/mpc/Dataset/OIQA/cubic/'
index = 1
img_path1 = root_path + str(index).rjust(3, '0') + 'F.jpg'
img_path2 = root_path + str(index).rjust(3, '0') + 'R.jpg'
img_path3 = root_path + str(index).rjust(3, '0') + 'BA.jpg'
img_path4 = root_path + str(index).rjust(3, '0') + 'L.jpg'
img_path5 = root_path + str(index).rjust(3, '0') + 'T.jpg'
img_path6 = root_path + str(index).rjust(3, '0') + 'BO.jpg'
img1 = Image.open(img_path1)
img2 = Image.open(img_path2)
img3 = Image.open(img_path3)
img4 = Image.open(img_path4)
img5 = Image.open(img_path5)
img6 = Image.open(img_path6)
img1 = img1.resize((256, 256))
img2 = img2.resize((256, 256))
img3 = img3.resize((256, 256))
img4 = img4.resize((256, 256))
img5 = img5.resize((256, 256))
img6 = img6.resize((256, 256))
save_root = './resized_cubic/'
if not os.path.exists(save_root):
    os.mkdir(save_root)
save_path1 = save_root + str(index).rjust(3, '0') + 'F.png'
save_path2 = save_root + str(index).rjust(3, '0') + 'R.png'
save_path3 = save_root + str(index).rjust(3, '0') + 'BA.png'
save_path4 = save_root + str(index).rjust(3, '0') + 'L.png'
save_path5 = save_root + str(index).rjust(3, '0') + 'T.png'
save_path6 = save_root + str(index).rjust(3, '0') + 'BO.png'
img1.save(save_path1)
img2.save(save_path2)
img3.save(save_path3)
img4.save(save_path4)
img5.save(save_path5)
img6.save(save_path6)
