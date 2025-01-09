#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
model = torch.hub.load('ultralytics/yolov5', 'custom', path = 'yolov5/runs/train/exp15/weights/last.pt', force_reload = True)


# In[3]:


import cv2
imag = 'C:\\Users\\UP2store\\Pictures\\p&hp\\IMG20240206231428.jpg'
image = cv2.imread(imag)


# In[4]:


results = model(image)


# In[5]:


results


# In[8]:


from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.imshow(np.squeeze(results.render()))
plt.show()


# In[9]:


results.xyxyn[0].numpy()


# In[10]:


import numpy as np

array = results.xyxyn[0].numpy()
fifth_elements = array[:, 5]  
print("Fifth elements in every list:", fifth_elements)

count_pound = 0
count_halfpound = 0

for element in fifth_elements:
    if element == 16:
        count_pound += 1
    else:
        count_halfpound += 1

total_amount = count_pound *1 + count_halfpound * 0.5
print(total_amount)

resized_input_image = cv2.resize(image, (800, 600))
output_image = resized_input_image.copy()

cv2.putText(output_image, f'Total money: {total_amount}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

cv2.imshow('Money Counter', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

