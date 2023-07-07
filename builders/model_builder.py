from model.ELANet import ELANet


def build_model(model_name, num_classes):
    if model_name == 'ELANet':
        return ELANet(classes=num_classes)
