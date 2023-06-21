from sklearn.model_selection import train_test_split
import os
import yaml
import shutil
import argparse

def create_train_test_val(config_path: str) -> None:
    data = yaml.safe_load(open(config_path,'r+'))
    raw_files_path = data["data_load"]["raw_images_path"]
    output_path = data["data_load"]["prepared_images_path"]
    label_path = os.path.join(raw_files_path,"labels")
    classes_file_source = os.path.join(raw_files_path,'classes.txt')

    # define the training modes for the dataset
    split_modes = ["train","val","test"]
    images = []
    labels = []

    # read the YOLO formatted data from raw dataset
    for dirpath,subdir, filelist in os.walk(raw_files_path):
        if dirpath == raw_files_path:
            continue
        # we only need one iteration
        if dirpath.endswith("labels"):
            continue
        for filename in filelist:
            label_name = filename.replace(".jpg",".txt")
            label_txt = os.path.join(label_path,label_name)
            image_path = os.path.join(dirpath,filename)
            if os.path.exists(label_txt) == True and \
                os.path.exists(image_path) == True:
                images.append(image_path)
                labels.append(label_txt)
            else:
                print (label_name,"Not Found in Preprocessing!")
                continue
            
            
            # print (label_txt, image_path)
    print ("Total Images:",len(images))
    images_train, images_split, labels_train, labels_split = train_test_split(
        images,
        labels,
        random_state=15,
        test_size=0.3
    )
    print ("Train Images:",len(images_train))
    images_val,images_test, labels_val,labels_test = train_test_split(
        images_split,
        labels_split,
        test_size=0.5,
        shuffle=True,
        random_state=15
    )
    print ("Validation Images:",len(images_val))
    print ("Test Images:",len(images_test))
    
    # consistent with split_mode 
    images_files_list = [images_train,images_val,images_test]
    labels_files_list = [labels_train,labels_val,labels_test]
    # empty everything in output dir then copy files
    for idx,mode in enumerate(split_modes):
        output_folder_path = os.path.join(output_path,mode)
        if os.path.exists(output_folder_path):
            shutil.rmtree(output_folder_path)
        os.mkdir(output_folder_path)
        os.mkdir(os.path.join(output_folder_path,'images'))
        os.mkdir(os.path.join(output_folder_path,'labels'))
        for j,source_image_path in enumerate(images_files_list[idx]):
            source_label_path = labels_files_list[idx][j]
            image_path = source_image_path.split("/")[-1]
            label_path = source_label_path.split("/")[-1]

            dest_image_path = os.path.join(output_folder_path,'images',image_path)
            dest_label_path = os.path.join(output_folder_path,'labels',label_path)
            # print (source_image_path,dest_image_path)
            shutil.copy(source_image_path,dest_image_path)
            shutil.copy(source_label_path,dest_label_path)

        ## Finally copy classes.txt to each data-split folders
        dest_classes_txt = os.path.join(output_folder_path,'classes.txt')
        shutil.copy(classes_file_source,dest_classes_txt)
    print ("Completed!!! ")

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config",dest='config',required=True)
    args = args_parser.parse_args()
    create_train_test_val(config_path=args.config)
