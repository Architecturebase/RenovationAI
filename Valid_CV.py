#!/usr/bin/env python
# coding: utf-8

# In[51]:


from ultralytics import YOLO
import torch


# In[93]:


source = r'C:\Users\user\Desktop\RenovationAI\monitoring\6.mp4'
model_path = r'C:\Users\user\Desktop\RenovationAI\weights\best.pt'


# In[8]:


model = YOLO(model_path)
model.predict(source, imgsz=640, conf=0.4, save=True)


# In[94]:


def model_predict(model_path, clip):
    model = YOLO(model_path)
    results = model(clip, imgsz = 640, conf = 0.4, save = True)
    return results


# In[10]:


results = model(source, imgsz = 640, conf = 0.4)


# In[79]:


import itertools
clean_boxes = []
for i in range(len(results)):
    if len(results[i].boxes.cls) != 0:
        clean_boxes.append(results[i].boxes.cls)
    else:
        continue
#         clean_boxes.append(torch.tensor([]))
        
ids = []
ids_flatten = []
for i in range(len(clean_boxes)):
    if len(clean_boxes[i]) <= 1:
        ids.append(int(clean_boxes[i].item()))
        ids_flatten.append(int(clean_boxes[i].item()))
    else:
        long_ids = []
        for k in range(len(clean_boxes[i])):
            long_ids.append(clean_boxes[i][k].item())
        long_ids = [int(x) for x in long_ids]
        ids.append(tuple(long_ids))
        
ids_sorted = [k for k,_g in itertools.groupby(ids)]


# In[82]:


a = [int(x) for x in range(38)]


# In[86]:


not_found = []
for i in range(len(a)):
    if a[i] not in ids:
        not_found.append(a[i])


# In[87]:


not_found


# In[89]:


#классы, которых нет на видео
for i in range(len(not_found)):
    print(names.get(not_found[i]))


# In[85]:


#все классы
names = {0: 'Balcony door', 1: 'Balcony long window', 2: 'Bathtub', 3: 'Battery', 4: 'Ceiling', 5: 'Chandelier', 6: 'Door', 7: 'Electrical panel', 8: 'Fire alarm', 9: 'Good Socket', 10: 'Gutters', 11: 'Laminatte', 12: 'Light switch', 13: 'Plate', 14: 'Sink', 15: 'Toilet', 16: 'Unfinished socket', 17: 'Wall tile', 18: 'Wallpaper', 19: 'Window', 20: 'Windowsill', 21: 'bare_ceiling', 22: 'bare_wall', 23: 'building_stuff', 24: 'bulb', 25: 'floor_not_screed', 26: 'floor_screed', 27: 'gas_blocks', 28: 'grilyato', 29: 'junk', 30: 'painted_wall', 31: 'pipes', 32: 'plastered_walls', 33: 'rough_ceiling', 34: 'sticking_wires', 35: 'tile', 36: 'unfinished_door', 37: 'unnecessary_hole'}


# In[78]:


ids


# In[76]:


ids_clean


# In[73]:


ids


# In[58]:


clean_boxes[4]


# In[53]:


clean_boxes


# In[ ]:




