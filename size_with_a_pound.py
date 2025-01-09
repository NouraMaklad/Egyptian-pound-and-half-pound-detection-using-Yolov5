#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')


# In[2]:


import cv2
img = 'C:\\Users\\UP2store\\Desktop\\IMG_20240212_011152.jpg'
imageb = cv2.imread(img)


# In[3]:


res = model(imageb)


# In[4]:


res


# In[6]:


from matplotlib import pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
plt.imshow(np.squeeze(res.render()))
plt.show()


# In[7]:


res.xywhn[0].numpy()


# In[8]:


res.xyxyn[0].numpy()


# In[9]:


model = torch.hub.load('ultralytics/yolov5', 'custom', path = 'yolov5/runs/train/exp15/weights/last.pt', force_reload = True)


# In[10]:


import cv2
image = cv2.imread(img)


# In[11]:


results = model(image)


# In[12]:


results


# In[13]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.imshow(np.squeeze(results.render()))
plt.show()


# In[14]:


results.xywhn[0].numpy()


# In[15]:


results.xyxyn[0].numpy()


# In[16]:


results.xywhn[0][0][2].numpy()


# In[18]:


bottle_coords = None
for detection in res.xyxy[0]:
    if int(detection[5]) == 39: 
        bottle_coords = detection[:4]
        break

if bottle_coords is not None:
    # Calculate width and height of the bottle
    bottle_width_pixels = res.xywhn[0][0][2].numpy()
    bottle_height_pixels = res.xywhn[0][0][3].numpy()

    # Dimensions of the pound in centimeters
    pound_width_cm = 2.5
    pound_height_cm = 2.5

    pound_width_pixels = results.xywhn[0][0][2].numpy()
    pound_height_pixels = results.xywhn[0][0][3].numpy()

    # Calculate scale for conversion
    width_scale = bottle_width_pixels / pound_width_pixels
    height_scale = bottle_height_pixels / pound_height_pixels

    # Calculate width and height of the bottle in centimeters
    bottle_width_cm = pound_width_cm * width_scale
    bottle_height_cm = pound_width_cm * height_scale
    
    resized_input_image = cv2.resize(image, (800, 600))
    output_image = resized_input_image.copy()

    # Write dimensions on the image
    cv2.putText(output_image, f'Width: {bottle_width_cm:.2f} cm', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(output_image, f'Height: {bottle_height_cm:.2f} cm', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Save the modified image
    cv2.imwrite('output_image.jpg', output_image)

    # Display the modified image
    cv2.imshow('Bottle Dimensions', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Bottle not detected in the image.")

