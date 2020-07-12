# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# imports and setup 

from pathlib import Path
import numpy as np 
import xml.etree.ElementTree as ET

from pillclassification.feature_extraction import feature_extraction

feature_number = 10
images_dir = Path('utils/Dataset/merge')
filenames = [x for x in images_dir.iterdir() if x.suffix != '.xml']


# %%
# calculating labels 
try:
    tree = ET.parse(images_dir / 'images.xml')
except ET.ParseError:
    print('Parse error on {}'.format(images_dir / 'images.xml'))
    exit(-1)

se = list(tree.getroot())[0]

labels_set = set()
for e in list(se):
    labels_set.add(e.find('NDC9').text)

labels = sorted(list(labels_set))


# %%
# extracting features
samples_num = len(filenames)
x_data = np.zeros((sample_num, feature_number))
y_data = np.zeros((samples_num))

print(x_data, y_data)

