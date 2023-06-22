import boto3
import argparse
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.debugger import Rule, rule_configs
from sagemaker.debugger import ProfilerRule
import os

AWS_SAGEMAKER_ROLE = os.environ.get('aws_sagemaker_role',None)



def train_sagemaker():
    sess = sagemaker.Session()
    bucket = sess.default_bucket()
    role = AWS_SAGEMAKER_ROLE
    region = sess.boto_region_name

    metric_definitions=[
        {
            "Name": "precision",
            "Regex": "YOLO Metric metrics/precision\(B\): (.*)"
        },
        {
            "Name": "recall",
            "Regex": "YOLO Metric metrics/recall\(B\): (.*)"
        },
        {
            "Name": "mAP50",
            "Regex": "YOLO Metric metrics/mAP50\(B\): (.*)"
        },
        {
            "Name": "mAP50-95",
            "Regex": "YOLO Metric metrics/mAP50-95\(B\): (.*)"
        },
        {
            "Name": "box_loss",
            "Regex": "YOLO Metric val/box_loss: (.*)"
        },
        {
            "Name": "cls_loss",
            "Regex": "YOLO Metric val/cls_loss: (.*)"
        },
        {
            "Name": "dfl_loss",
            "Regex": "YOLO Metric val/dfl_loss: (.*)"
        }
    ]



    rules = [
        # can have debugging and profiling rules
        Rule.sagemaker(rule_configs.loss_not_decreasing()),
        Rule.sagemaker(rule_configs.overtraining()),
        ProfilerRule.sagemaker(rule_configs.LowGPUUtilization()),
        ProfilerRule.sagemaker(rule_configs.ProfilerReport())
    ]


    # input
    estimator = PyTorch(
        entry_point='train.py',
        source_dir="src",
        py_version="py310",
        framework_version="2.0",
        role=role,
        instance_count = 1,
        instance_type='ml.g4dn.xlarge',
        use_spot_insances=True,
        output_path=f"s3://{bucket}/model_artifacts/",
        metric_definitions=metric_definitions,
        rules=rules
    )

    estimator.fit()


    return


if __name__  == '__main__':
    arg_parse = argparse.ArgumentParser()
    train_sagemaker()