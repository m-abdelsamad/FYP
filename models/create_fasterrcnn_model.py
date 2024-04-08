from models import *

def return_fasterrcnn_resnet50(
    num_classes, pretrained=True,
):
    model = fasterrcnn_resnet50.create_model(
        num_classes, pretrained=pretrained,
    )
    return model

def return_fasterrcnn_resnet50_fpn(
    num_classes, pretrained=True,
):
    model = fasterrcnn_resnet50_fpn.create_model(
        num_classes, pretrained=pretrained,
    )
    return model

def return_fasterrcnn_resnet50_cbam_w_aspp(
    num_classes, pretrained=True,
):
    model = fasterrcnn_resnet50_cbam_w_aspp.create_model(
        num_classes, pretrained=pretrained,
    )
    return model

create_model = {
    'fasterrcnn_resnet50_fpn': return_fasterrcnn_resnet50_fpn,
    'fasterrcnn_resnet50': return_fasterrcnn_resnet50,
    'fasterrcnn_resnet50_cbam_w_aspp': return_fasterrcnn_resnet50_cbam_w_aspp
}