import segmentation_models_pytorch as smp

from segwork.registry import ConfigurableRegistry

__all__ = ['backbones_reg', 'models_reg']
_initial_model_registry = dict()

try:
    import segmentation_models_pytorch as smp

    def smp_hook(key, value):
        """Safe inclusion of key and value into smp encoders repository"""
        smp.encoders.encoders[key] = value

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

    _initial_backbone_registry = smp.encoders.encoders
    _register_hook = smp_hook # Retrocompatibility
    _default_kwargs = 'params'
except Exception as e:
    # loggin.warning(f'segmentation_pytorch not installed.')
    print(e)
    _initial_backbone_registry = dict()
    _initial_model_registry = dict()
    _register_hook = None
    _default_kwargs = '_default_kwargs'

backbones_reg = ConfigurableRegistry(
    class_key = 'encoder',                      # Key to the nn.module class
    initial_registry = _initial_backbone_registry,       # Initial registry. Default: None
    attr_args = _default_kwargs,
    additional_args= ['pretrained_settings'],
    register_hook= _register_hook) 

models_reg = ConfigurableRegistry(
    class_key = 'model',                      # Key to the nn.module class
    initial_registry = _initial_model_registry )