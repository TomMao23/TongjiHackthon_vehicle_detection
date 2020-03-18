
import os
import cv2
import xml.etree.ElementTree as ET
from tqdm import tqdm
classes = ['Truck','Car','Van','car','bus', 'truck']
input_path = '/media/tommao/F/data/highway_vehicle/Annotations1/'
output_path = '/media/tommao/F/data/highway_vehicle/Annotations/'

def purifyAnnotations(input_path,output_path,classes):
    """
    入口参数：
    input_path是需要处理的释文文件夹路径，output_path是生成的xml存放的文件夹路径，classes是需要保留的目标类的数组，e.g. classes=["Truck","Car"]
    
    出口参数：
    返回一个包含所有缺少目标类的xml的编号的数组
    """
    print(output_path)#确定路径
    annots_list=[]
    
    annots = [s for s in os.listdir(input_path)]#所有xml文件名
#     annots = tqdm(annots)#用来可视化进度
    
    for annot in annots:#遍历所有xml文件
#         annots.set_description(annot)#增加进度条可读性

        write_path = os.path.join(output_path, annot)#预先留下生成路径
        annot_path = os.path.join(input_path, annot)#xml完整路径
        
        et = ET.parse(annot_path)#打开xml文件
        element = et.getroot()#获得root节点
        
        element_objs = element.findall('object')#找到所有object元素
        for j in element_objs:#遍历每个object元素
            if j.find('name').text not in classes:#如果不在我们的目标类中
                element.remove(j)#移除这个object元素
                
        objs_count = len(element.findall('object'))#更新余下object元素个数
#         if objs_count == 0:#说明该xml没有我们的目标类
        if objs_count != 0:#说明该xml有我们的目标类
#             annots_list.append(annot)#加入名单！
            annots_list.append(annot.split('.')[0])#xml的编号加入名单！
            
        write_path = os.path.join(output_path, annot)
        et.write(write_path)#生成修改之后的单个xml文件
        
    txt_path = os.path.join(output_path, 'list.txt')
    with open('list.txt',"w+") as f:    #设置文件对象，若不存在则创建，若存在则会覆盖重写！！
        for annot_list in annots_list:
            f.write(annot_list+'\n')

    return annots_list
    
if __name__ == '__main__':
    purifyAnnotations(input_path,output_path,classes)   
