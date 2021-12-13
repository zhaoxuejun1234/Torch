import torch
import numpy as np
import torch.nn.functional as F
import cv2
from PIL import Image
import matplotlib.pyplot as plt
# a = np.zeros((100,100,3),np.float32)
# # print(a)
# # print(np.array.__doc__)
# b = np.random.randn(3,3)
# bb = np.random.random(3)
# print(bb)
# # print(b)
# # print(b.argmax(axis=0))
# # print(b.argmax(axis=1))
#
# c=np.zeros((3,3),dtype=np.int32)
# # print(c.dtype)
# d=np.zeros((3,3)).astype(np.float32)
# # print(d.dtype)
# # print(d.shape)
# e = d.copy()
# # print(d-e)
#
#
# f= np.zeros((3,3)).astype(np.float32)
#
# gg=np.zeros((3,3),dtype=np.float32)
#
# cv2.imread()


a = torch.ones(3,3).float()
# print(a)


b = torch.ones((3,3),dtype=torch.float32)
# print(b)


c=torch.rand(3,3)
d= np.random.randn(3,3)

e =d[None,:,:,None]
# print(e.shape)




a=torch.tensor(10.).requires_grad_(True)
b=torch.tensor(5.,requires_grad=True)
# print(a,b)

f = torch.randn(2,3)
print(f)







