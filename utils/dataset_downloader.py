from ftplib import FTP
import utils
from pathlib import Path
import xml.etree.ElementTree as ET
import glob

server='lhcftp.nlm.nih.gov'
dirs_from_to=(1, 110)
source_dirs=['Open-Access-Datasets/Pills/PillProjectDisc' + str(x) +'/images' for x in range(dirs_from_to[0], dirs_from_to[1]+1)]

dest_base_dir=Path('/media/mirko/Mirko HDD/Dataset')

with FTP(server) as ftp:
    print('Login into {}'.format(server))
    ftp.login()

    base_dir = ftp.pwd()
    
    for idx, source_dir in enumerate(source_dirs):
        ftp.cwd(base_dir)
        ftp.cwd(source_dir)
        
        dest_dir = dest_base_dir / str(idx)
        
        dest_dir.mkdir(parents=True, exist_ok=True)

        print('Current woring directory:', ftp.pwd())
        print('Downloading tmp xml')
        filename = 'images.xml'
        dest_file = dest_dir / filename
        with open(dest_file, 'wb') as f:
            ftp.retrbinary('RETR ' + filename, f.write)

        tree = ET.parse(dest_file)
        root = tree.getroot()
        
        se = list(root)[0]
        
        print('Filtering xml (' + str(len(list(se))) + ' images)')
        images = []
        for e in list(se):
            layout = e.find('Layout')
            shadow = e.find('RatingShadow')
            colors = e.findall('Color')
            if (layout is not None and layout.text == "MC_C3PI_REFERENCE_SEG_V1.6") or \
               (shadow is not None and shadow.text == 'Soft') and \
               (colors is not None and len(colors) == 1):
                images.append(e.find('File').find('Name').text)
            else:
                se.remove(e)
        
        print('saving xml in:', dest_file)
        tree.write(dest_file)
        
        print("final images:", len(images))


dirs = [x for x in dest_base_dir.iterdir() if x.is_dir()]

ids = dict()

expected_size = 0

for d in dirs:
    try:
        tree = ET.parse(d / 'images.xml')
    except ET.ParseError:
        print('Parse error on {}'.format(d/'images.xml'))
        continue
    se = list(tree.getroot())[0]
    
    for e in list(se):
        expected_size += int(e.find('File').find('Size').text)
        
        # i = e.find('ProprietaryName').text.lower()
        # i = e.find('NDC11').text[5:9]
        i = e.find('NDC9').text
        if i not in ids:
            ids[i] = []
        ids[i].append(e.find('File').find('Name').text) 

sizes = dict()

for k, e in ids.items():
    if len(e) not in sizes:
        sizes[len(e)] = []
    sizes[len(e)].append(k)


sorted_sizes = list(sizes.keys())
sorted_sizes.sort(reverse=True)

total_ids = 0
total_images = 0
for k in sorted_sizes:
    print('{:4} ids with {:4} images'.format(len(sizes[k]), k))
    total_ids += len(sizes[k])
    total_images += len(sizes[k]) * k
    
print('Total ids: {}'.format(total_ids))
print('Total images: {}'.format(total_images))

print(utils.bytes2human(expected_size), 'will be needed to download all the images')


a = input('> ')
ids_to_download = sizes[a]
download_imgs = True

with FTP(server) as ftp:
    print('Login into {}'.format(server))
    ftp.login()

    base_dir = ftp.pwd()
    
    for idx, source_dir in enumerate(source_dirs):
        ftp.cwd(base_dir)
        ftp.cwd(source_dir)
        
        dest_dir = dest_base_dir / str(idx)
        
        print(dest_dir)
        
        try:
            tree = ET.parse(dest_dir / 'images.xml')
        except ET.ParseError:
            print('Parse error on {}'.format(dest_dir / 'images.xml'))
            continue
        se = list(tree.getroot())[0]
        
        images = []
        
        for e in list(se):
            ndc = e.find('NDC9').text
            if ndc in ids_to_download:
                images.append(e.find('File').find('Name').text)
        
        # downloading
        if download_imgs:
            for i, img in enumerate(images):
                dest_file = dest_dir / img
                if (dest_file).exists():
                    print(img, 'already downloaded!')
                    continue
                with open(dest_file, 'wb') as f:
                    print('\rDownloading {:3}/{}'.format(i + 1, len(images)), end='')
                    ftp.retrbinary('RETR ' + img, f.write)
            print()


# Merge in one dir

import shutil

merge_dir = dest_base_dir / 'merge'

merge_dir.mkdir(parents=True, exist_ok=True)

merge_xml = ET.Element('MedicosConsultants')
ET.SubElement(merge_xml, 'ImageExport')

for i, d in enumerate(dirs):
    print('\rMerging dir {:3}/{}'.format(i + 1, len(dirs)), end='')
    imgs = [x.name for x in d.iterdir() if x.suffix != '.xml']
    
    tree = ET.parse(d / 'images.xml')
    
    se = list(tree.getroot())[0]
    
    for e in list(se):
        name = e.find('File').find('Name').text
        if name in imgs:
            merge_xml[0].append(e)
    
    for img in imgs:
        shutil.move(str(d / img), str(merge_dir))

print('\nSaving xml')
ET.ElementTree(merge_xml).write(merge_dir / 'images.xml')
