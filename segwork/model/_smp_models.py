import segmentation_models_pytorch as smp

_initial_model_registry = dict()

unet_registry = dict(
    model = smp.Unet,
    params = {
        'encoder_name':  "resnet34",
        'encoder_depth':  5,
        'encoder_weights':"imagenet",
        'decoder_use_batchnorm':True,
        'decoder_channels': (256, 128, 64, 32, 16),
        'decoder_attention_type': None,
        'in_channels':  3,
        'classes': 1,
        'activation': None,
        'aux_params':  None,
    }
)

unetplusplus_registry = dict(
    model = smp.UnetPlusPlus,
    params = {
        'encoder_name':  "resnet34",
        'encoder_depth':  5,
        'encoder_weights':"imagenet",
        'decoder_use_batchnorm':True,
        'decoder_channels': (256, 128, 64, 32, 16),
        'decoder_attention_type': None,
        'in_channels':  3,
        'classes': 1,
        'activation': None,
        'aux_params':  None,
    }
)

manet_registry = dict(
    model = smp.MAnet,
    params = {
        'encoder_name':  "resnet34",
        'encoder_depth':  5,
        'encoder_weights':"imagenet",
        'decoder_use_batchnorm':True,
        'decoder_channels': (256, 128, 64, 32, 16),
        'decoder_pab_channels': 64,
        'in_channels':  3,
        'classes': 1,
        'activation': None,
        'aux_params':  None,
    }
)

linknet_registry = dict(
    model = smp.Linknet,
    params = {
        'encoder_name':  "resnet34",
        'encoder_depth':  5,
        'encoder_weights':"imagenet",
        'decoder_use_batchnorm':True,
        'in_channels':  3,
        'classes': 1,
        'activation': None,
        'aux_params':  None,
    }
)

fpn_registry = dict(
    model = smp.FPN,
    params = {
        'encoder_name':  "resnet34",
        'encoder_depth':  5,
        'encoder_weights':"imagenet",
        # 'decoder_pyramid_channels':True,
        # 'decoder_segmentation_channels':True,
        # 'decoder_merge_policy':True,
        'in_channels':  3,
        'classes': 1,
        'activation': None,
        'aux_params':  None,
    }

)

psp_registry = dict(
    model = smp.PSPNet,
    params = {
        'encoder_name':  "resnet34",
        'encoder_depth':  5,
        'encoder_weights':"imagenet",
        # 'psp_out_channels':True,
        'in_channels':  3,
        'classes': 1,
        'activation': None,
        'aux_params':  None,
    }
)

pan_registry = dict(
    model = smp.PAN,
    params = {
        'encoder_name': "resnet34",
        'encoder_depth': 5,
        'encoder_weights':"imagenet",
        'encoder_output_stride' : 16,
        'in_channels':  3,
        'classes': 1,
        'activation': None,
        'aux_params':  None,
    }
)

deeplabv3_registry = dict(
    model = smp.DeepLabV3,
    params = {
        'encoder_name': "resnet34",
        'encoder_depth': 5,
        'encoder_weights':"imagenet",
        'decoder_channels' : 256,
        'in_channels':  3,
        'classes': 1,
        'activation': None,
        'aux_params':  None,
    }
)
deeplabv3plus_registry = dict(
    model = smp.DeepLabV3Plus,
    params = {
        'encoder_name': "resnet34",
        'encoder_depth': 5,
        'encoder_weights':"imagenet",
        'decder_channels': 256,
        # 'encoder_output_stride' : 256,
        # 'decoder_atrous_rates' : 256,
        'in_channels':  3,
        'classes': 1,
        'activation': None,
        'aux_params':  None,
    }
)





_initial_model_registry['unet'] = unet_registry
_initial_model_registry['unet++'] = unetplusplus_registry
_initial_model_registry['manet'] = manet_registry
_initial_model_registry['linknet'] = linknet_registry
_initial_model_registry['fpn'] = fpn_registry
_initial_model_registry['psp'] = psp_registry
_initial_model_registry['pan'] = pan_registry
_initial_model_registry['deeplabv3'] = deeplabv3_registry
_initial_model_registry['deeplabv3plus'] = deeplabv3_registry