import segmentation_models_pytorch as smp

def get_model(arch="resnet18", in_channels=192, num_classes=7):
    """
    Return the requested segmentation model (UNet with chosen encoder).
    Encoders examples: resnet18, resnet34, resnet50, mobilenet_v2, efficientnet_b5.
    """
    if arch in ["resnet18", "resnet34", "resnet50"]:
        model = smp.Unet(
            encoder_name=arch,
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes
        )
    elif arch.startswith("mobilenet") or arch.startswith("efficientnet"):
        model = smp.Unet(
            encoder_name=arch,
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes
        )
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    return model

