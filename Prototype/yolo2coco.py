from imports import *


def yolotxt2cocojson(save_dir, class_txt_path, label_dir, img_dir):
    with open(class_txt_path,'r') as fr: #Open and read category files
        lines1 = fr. readlines()
    # print(lines1)
    categories=[] #list of storage categories
    for j,label in enumerate(lines1):
        label = label. strip()
        categories.append({'id':j+1,'name':label,'supercategory':'None'}) #Add category information to categories
    # print(categories)

    write_json_context=dict() #Write a large dictionary of .json files
    write_json_context['info']= {'description': '', 'url': '', 'version': '', 'year': 2023 , 'contributor': 'pure ss', 'date_created': '2023-10-9'}
    write_json_context['licenses']=[{'id':1,'name':None,'url':None}]
    write_json_context['categories']=categories
    write_json_context['images']=[]
    write_json_context['annotations']=[]

    #The following code mainly adds the key values of 'images' and 'annotations'
    img_list=os.listdir(img_dir) #traverse all files in this folder, and add all file names to the list
    for i,img_name in enumerate(img_list):
      if img_name[-4:] == ".png":
        img_path = f"{img_dir}/{img_name}" #Get the absolute path of the image
        image = cv2.imread(img_path) #Read the picture, then get the width and height of the picture
        H, W = image.shape[:2]
        # print(img_name)
        # print(type(img_name))

        img_context={} #Use a dictionary to store the image information
        #img_name=os.path.basename(img_path) #Return the last file name of path. If path ends with / or \, an empty value will be returned
        img_context['file_name']=img_name
        # print(img_context['file_name'])
        img_context['height']=H
        img_context['width']=W
        img_context['date_captured']='2023-10-9'
        img_context['id']=i #The id of the picture
        img_context['license']=1
        img_context['color_url']=''
        img_context['flickr_url']=''
        write_json_context['images'].append(img_context) #Add the image information to the 'image' list
        # print(img_name)

        label_name = img_name.split('.')[0] + '.txt' #Get the txt file corresponding to the picture
        with open(f"{label_dir}/{label_name}",'r') as fr:
            lines=fr.readlines() #Read each line of data in the txt file, lines2 is a list that contains all the annotation information of a picture
        for j,line in enumerate(lines):

            bbox_dict = {} #store each bounding box information in the dictionary
            # line = line.strip().split()
            # print(line. strip(). split(' '))

            class_id,x,y,w,h=line.strip().split(' ') #Get the detailed information of each label box
            class_id,x, y, w, h = int(class_id), float(x), float(y), float(w), float(h) #Convert string type to computable int and float type

            xmin=(x-w/2)*W #coordinate conversion
            ymin=(y-h/2)*H
            xmax=(x + w/2)*W
            ymax=(y + h/2)*H
            w=w*W
            h=h*H

            bbox_dict['id']=i*10000 + j #bounding box coordinate information
            bbox_dict['image_id']=i
            bbox_dict['category_id']=class_id + 1 #Note that the target category should be added by one
            bbox_dict['iscrowd']=0
            height,width=abs(ymax-ymin),abs(xmax-xmin)
            bbox_dict['area']=height*width
            bbox_dict['bbox']=[xmin,ymin,w,h]
            bbox_dict['segmentation']=[[xmin,ymin,xmax,ymin,xmax,ymax,xmin,ymax]]
            write_json_context['annotations'].append(bbox_dict) #Add each bounding box information stored by the dictionary to the 'annotations' list
    name = os.path.join(save_dir+'/images','_annotations.coco.json')
    with open(name,'w') as fw: #Write the dictionary information into the .json file
        json.dump(write_json_context,fw,indent=2,ensure_ascii=False) #Add suffix ensure_ascii=False to place Chinese garbled characters after writing


def the_number_of_file_in_dir(img_dir, label_dir):
    # Check the number of files in the two folders
    label_list = os.listdir(label_dir)
    img_list = os.listdir(img_dir)
    print(len(label_list))
    print(len(img_list))


def create_coco_file(save_dir, class_txt_path, label_dir, img_dir):
    # Generate training set file
    save_dir = os.path.join(save_dir)
    label_dir = os.path.join(label_dir)
    img_dir = os.path.join(img_dir)
    # the_number_of_file_in_dir(img_dir,label_dir)
    yolotxt2cocojson(save_dir, class_txt_path, label_dir, img_dir)
