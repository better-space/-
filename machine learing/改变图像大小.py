# from PIL import Image
# import matplotlib.image as mpimg
# import matplotlib.pyplot as plt
#
# mpimg_img = mpimg.imread('before.jpg')
# # plt.imshow(mpimg_img)
# # plt.show()
#
# # img = Image.open('before.jpg')
# # print(type(img))
# img2 = mpimg_img.squeeze((150,200))
# print(type(img2))
# img2.show() # 可以先预览一下
# # img2.save('now.jpg')

from PIL import Image
import numpy as np
img = Image.open('before.jpg')
# print(img.shape)
img2 = img.resize((200,150))
array = np.array(img2)
print(array.shape)
img2.show() # 可以先预览一下
img2.save('now.jpg')