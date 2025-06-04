"""
To select the architecture based on a config file we need to ensure
we import each of the architectures into this file. Once we have that
we can use a keyword from the config file to build the model.
"""


######################################################################
def build_architecture(config):
    if config["model_name"] == "mobileunetr_xs":
        from .mobileunetr import build_mobileunetr_xs

        model = build_mobileunetr_xs(config=config)
        return model

    elif config["model_name"] == "mobileunetr_xxs":
        from .mobileunetr import build_mobileunetr_xxs

        model = build_mobileunetr_xxs(config=config)
        return model

    elif config["model_name"] == "mobileunetr_s":
        from .mobileunetr import build_mobileunetr_s

        model = build_mobileunetr_s(config=config)
        return model
    
    elif config["model_name"] == "efficientnet_v2_s_unet":
        from .efficientnet_unet import build_efficientnet_v2_s_unet
        
        model = build_efficientnet_v2_s_unet(
            num_classes=config["variables"]["num_classes"],
            pretrained=True
        )
        return model

    else:
        return ValueError(
            "specified model not supported, edit build_architecture.py file"
        )
