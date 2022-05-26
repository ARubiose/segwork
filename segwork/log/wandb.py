try:
    import wandb
except ImportError:
    raise ImportError('wandb library must be installed to use the wandb logger')

class WandbLogger(object):
    """Class to save logs on Wandb"""
    def __init__(self, project_name: str, entity: str, config={}):
        self._run = wandb.init(project=project_name, entity=entity, config=config)

    def write_hyper(self, args):
        wandb.config.update(args, allow_val_change=True)

    def write_logs(self, args):
        wandb.log( args )

    def watch_model(self, model):
        wandb.watch(
        model, criterion=None, log="all", log_freq=1000, idx=None,
        log_graph=(False)
    )

    def get_config_dict(args):
        hyperparameter_list = ['freeze', 'batch_size', 'height', 'width', 'epochs', 'lr', 'weight_decay' \
            'momentum', 'optimizer', 'class_weighting', 'he_init', 'activation', 'encoder', 'encoder_block' \
            'nr_decoder_blocks', 'context_module', 'channels_decoder', 'decoder_channels_mode', 
            'upsampling', 'dataset', 'workers']

        return hyperparameter_list