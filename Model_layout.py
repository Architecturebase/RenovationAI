#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from roboflow import Roboflow
rf = Roboflow(api_key="")
project = rf.workspace().project("layout_detection")
model = project.version(1).model

