from ultralytics import YOLO
import torch
import yaml
import argparse
from datetime import datetime
import shutil
import os,gc
import pickle
from numba import cuda


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

    # empty the cache just in case
    model = None
    device = cuda.get_current_device()
    device.reset()
    torch.cuda.empty_cache()
    gc.collect()
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
    model_saved_path = str(model.metrics.save_dir)
    model_args = model.ckpt["train_args"]
    model_metrics = model.metrics.__dict__
    del model_metrics["on_plot"]
    
    test_metrics = model.val(split='test')
    test_metrics = test_metrics.__dict__
    del test_metrics["on_plot"]

    model_data = {}
    model_data["model_name"] = model_path
    model_data["val_metrics"] = model_metrics
    model_data["test_metrics"] = test_metrics
    model_data["train_args"] = model_args
    pickle.dump(
        model_data,
        open(
            os.path.join(model_saved_path,"metrics.pickle"),
            "wb"
        ),
        protocol=pickle.HIGHEST_PROTOCOL
    )
    

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config",dest='config',default="params.yaml")
    args = args_parser.parse_args()
    print ("Starting the training job!~~~~~~")
    train(config_path=args.config)
    print ("Completed!!!")
