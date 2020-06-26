from ftplib import FTP
import utils
import os
import fire
import xml.etree.ElementTree as ET
import glob


def dataset_downloader(server='lhcftp.nlm.nih.gov',
        source_dirs=['Open-Access-Datasets/Pills/PillProjectDisc1/images'],
        dest_dir='../Dataset/consumer',
        max_file_size='1M',
        max_files_per_dir=10):
    if max_file_size is not None:
        max_file_size = utils.human2bytes(max_file_size)

    with FTP(server) as ftp:
        print('Login into {}'.format(server))
        ftp.login()

        base_dir = ftp.pwd()
        
        for source_dir in source_dirs:
            i = max_files_per_dir
            ftp.cwd(base_dir)
            ftp.cwd(source_dir)

            print('Downloading images from {}'.format(ftp.pwd()))

            images = []

            nlst = ftp.nlst()
            for filename in nlst:
                size = ftp.size(filename)
                if max_file_size is not None and size > max_file_size:
                    continue
                print('{}: {:>10}'.format(filename, utils.bytes2human(size)))
                with open(os.path.join(dest_dir, filename), 'wb') as f:
                    ftp.retrbinary('RETR ' + filename, f.write)
                    images.append(filename)
                if i is not None and i > 1:
                    i -= 1
                else:
                    break
            
            print('Downloading tmp xml')
            filename = 'images.xml'
            with open(os.path.join(filename), 'wb') as f:
                ftp.retrbinary('RETR ' + filename, f.write)

            tree = ET.parse(filename)
            root = tree.getroot()
            
            se = list(root)[0]
            
            print('Filtering xml')
            for e in list(se):
                if e.find('.//Name').text not in images:
                    se.remove(e)

            print('Saving filtered xml')
            tree.write(os.path.join(dest_dir, filename))

if __name__ == '__main__':
    fire.Fire(dataset_downloader)
