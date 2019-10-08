from models_2D.resnet import resnet18, resnet34, resnet50


backbone_dict = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
}