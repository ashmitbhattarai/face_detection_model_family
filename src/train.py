from ultralytics import YOLO
import yaml
import argparse
from datetime import datetime
import shutil
import os


def train(config_path: str) -> None:
    data = yaml.safe_load(open(config_path,'r+'))
    model_path = data["train"]["pretrained_path"]
    dataset_path = data["train"]["dataset_path"]

    hyperparams = data["train"]["hyperparams"]
    project_name = "face_detection_model"
    project_path = os.path.join('runs',project_name)
    
    #empty the folder then create the path
    if os.path.exists(project_name):    
        shutil.rmtree(project_name)
        shutil.rmtree('wandb')
        os.mkdir(project_name)
    
    experiment_name = str(model_path.split("/")[-1])\
        + "_" + datetime.now().strftime("%Y_%m_%d-%H_%M")
    experiment_name = experiment_name.replace(".","_")

    print (project_name,experiment_name)
    print ("Clearing the Model Artifacts Path")
    # model = "" # comment it out later or delte it
    print (model_path,"model path")
    print ("dataset PATH",dataset_path)
    model = YOLO(model_path)

    model.train(

        data=dataset_path,
        epochs = hyperparams["epochs"],
        imgsz = hyperparams["img_size"],
        batch = hyperparams["batch_size"],
        workers = hyperparams["workers"],
        agnostic_nms = hyperparams["agnostic_nms"],
        conf = hyperparams["conf"],
        project = project_name,
        name=experiment_name
    )
    # training is completed, save the metrics and models
    model_metrics = model.metrics
    path_saved_model = str(model_metrics.save_dir)
    # model_dest = 

    # path_to_confusion_matrix_plot = os.path.join(project_path,experiment_name,)



    # test_result = model.val(split='test')
    
    print ("Completed")

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config",dest='config',required=True)
    args = args_parser.parse_args()
    train(config_path=args.config)