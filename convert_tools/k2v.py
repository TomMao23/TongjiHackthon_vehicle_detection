#!/usr/bin/env python
# -*- coding: utf8 -*-
import sys
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
import codecs
import os
import cv2

XML_EXT = '.xml'
ENCODE_METHOD = 'utf-8'

class PascalVocWriter:

    def __init__(self, foldername, filename, imgSize,databaseSrc='Unknown', localImgPath=None):
        self.foldername = foldername
        self.filename = filename
        self.databaseSrc = databaseSrc
        self.imgSize = imgSize
        self.boxlist = []
        self.localImgPath = localImgPath
        self.verified = False

    def prettify(self, elem):
        """
            Return a pretty-printed XML string for the Element.
        """
        rough_string = ElementTree.tostring(elem, 'utf8')
        root = etree.fromstring(rough_string)
        return etree.tostring(root, pretty_print=True, encoding=ENCODE_METHOD).replace("  ".encode(), "\t".encode())
        # minidom does not support UTF-8
        '''reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="\t", encoding=ENCODE_METHOD)'''

    def genXML(self):
        """
            Return XML root
        """
        # Check conditions
        if self.filename is None or \
                self.foldername is None or \
                self.imgSize is None:
            return None

        top = Element('annotation')
        if self.verified:
            top.set('verified', 'yes')

        folder = SubElement(top, 'folder')
        folder.text = self.foldername

        filename = SubElement(top, 'filename')
        filename.text = self.filename

        if self.localImgPath is not None:
            localImgPath = SubElement(top, 'path')
            localImgPath.text = self.localImgPath

        source = SubElement(top, 'source')
        database = SubElement(source, 'database')
        database.text = self.databaseSrc

        size_part = SubElement(top, 'size')
        width = SubElement(size_part, 'width')
        height = SubElement(size_part, 'height')
        depth = SubElement(size_part, 'depth')
        width.text = str(self.imgSize[1])
        height.text = str(self.imgSize[0])
        if len(self.imgSize) == 3:
            depth.text = str(self.imgSize[2])
        else:
            depth.text = '1'

        segmented = SubElement(top, 'segmented')
        segmented.text = '0'
        return top

    def addBndBox(self, xmin, ymin, xmax, ymax, name, difficult):
        bndbox = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
        bndbox['name'] = name
        bndbox['difficult'] = difficult
        self.boxlist.append(bndbox)

    def appendObjects(self, top):
        for each_object in self.boxlist:
            object_item = SubElement(top, 'object')
            name = SubElement(object_item, 'name')
            try:
                name.text = unicode(each_object['name'])
            except NameError:
                # Py3: NameError: name 'unicode' is not defined
                name.text = each_object['name']
            pose = SubElement(object_item, 'pose')
            pose.text = "Unspecified"
            truncated = SubElement(object_item, 'truncated')
            if int(each_object['ymax']) == int(self.imgSize[0]) or (int(each_object['ymin'])== 1):
                truncated.text = "1" # max == height or min
            elif (int(each_object['xmax'])==int(self.imgSize[1])) or (int(each_object['xmin'])== 1):
                truncated.text = "1" # max == width or min
            else:
                truncated.text = "0"
            difficult = SubElement(object_item, 'difficult')
            difficult.text = str( bool(each_object['difficult']) & 1 )
            bndbox = SubElement(object_item, 'bndbox')
            xmin = SubElement(bndbox, 'xmin')
            xmin.text = str(each_object['xmin'])
            ymin = SubElement(bndbox, 'ymin')
            ymin.text = str(each_object['ymin'])
            xmax = SubElement(bndbox, 'xmax')
            xmax.text = str(each_object['xmax'])
            ymax = SubElement(bndbox, 'ymax')
            ymax.text = str(each_object['ymax'])

    def save(self, targetFile=None):
        root = self.genXML()
        self.appendObjects(root)
        out_file = None
        if targetFile is None:
            out_file = codecs.open(
                self.filename + XML_EXT, 'w', encoding=ENCODE_METHOD)
        else:
            out_file = codecs.open(targetFile, 'w', encoding=ENCODE_METHOD)

        prettifyResult = self.prettify(root)
        out_file.write(prettifyResult.decode('utf8'))
        out_file.close()

class kitti2voc:
    """
        transfer the kitti dataset to voc dataset ,erase the "donot care" part of original images 
    """
    def __init__(self,image_path_src , image_path_dst , kitti_anat_path , voc_anat_path):
        """
            indicate the path 
        """
        self.image_path_src = image_path_src
        self.image_path_dst = image_path_dst
        self.kitti_anat_path = kitti_anat_path
        self.voc_anat_path   = voc_anat_path
    def convert_process(self):
        """
            to transfer the anat
        """
        #read the anat 
        filter_item = ["DontCare"]
        for filename in os.listdir(self.kitti_anat_path):
            print(filename)
            image_name = filename.replace(".txt",".png")
            image_file = image_path_src+image_name
            img = cv2.imread(image_file)
            image_shape=img.shape#a tuple
            #define the xml instance
            PascalVocWriter_ins = PascalVocWriter(self.voc_anat_path,image_name,image_shape)
            count = 0 
            kitti_ant = open(self.kitti_anat_path+filename,'r')
            for line in kitti_ant:
                item_list = line.split(' ')
                if item_list[0] not in filter_item :
                    count += 1 
                    PascalVocWriter_ins.addBndBox(int(float(item_list[4])),int(float(item_list[5])),int(float(item_list[6])),int(float(item_list[7])),item_list[0],0)
                else:
                    #pass
                    img[int(float(item_list[5])):int(float(item_list[7])),int(float(item_list[4])):int(float(item_list[6])),:] =0
                    #data = "the coordinate of anchor xm{},xma{},ym{},yma{}".format(int(float(item_list[4])),int(float(item_list[6])),int(float(item_list[5])),int(float(item_list[7])))
                    #print(data)
                    #img[503:590,169:190,:]=255
                    #print('imhere') 

            if count>0:
                PascalVocWriter_ins.save(voc_anat_path+'0'+filename.replace(".txt",".xml"))
                cv2.imwrite((image_path_dst+'0'+image_name)[:-3]+"jpg",img)


if __name__ =='__main__':
    #you must change the path  to yourselves ,and run 'python convert_k_x.py',the python version is 3.6 
    image_path_src = "/media/tommao/F/data/data_object_image_2/training/image_2/"
    image_path_dst = "/media/tommao/F/data/highway_vehicle/JPEGImages/"
    kitti_anat_path= "/media/tommao/F/data/training/label_2/"
    voc_anat_path = "/media/tommao/F/data/highway_vehicle/Annotations1/"
    kitti2voc_ins = kitti2voc(image_path_src,image_path_dst,kitti_anat_path,voc_anat_path)
    kitti2voc_ins.convert_process()
    print ("convert over")
    
    
    
    
    
    
    
    
