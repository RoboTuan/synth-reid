import sys
sys.path.append(".")
import time
from devreminder import DevReminder
import os.path as osp
import argparse
import torch
import torch.nn as nn
import torchreid
import wandb
import json

# remind = DevReminder(#########)
# remind.me("experiment_name")

from torchreid.utils import (
    Logger, check_isfile, set_random_seed, collect_env_info,
    resume_from_checkpoint, load_pretrained_weights, compute_model_complexity
)

from default_config import (
    imagedata_kwargs, optimizer_kwargs, generator_optimizer_kwargs, discriminator_optimizer_kwargs,
    lr_scheduler_kwargs, generator_lr_scheduler_kwargs, discriminator_lr_scheduler_kwargs,
    videodata_kwargs, engine_run_kwargs, get_default_config,
)


def make_datamanager(cfg):
    if cfg.data.type == 'image':
        return torchreid.data.ImageDataManager(**imagedata_kwargs(cfg))
    else:
        return torchreid.data.VideoDataManager(**videodata_kwargs(cfg))


def make_model(cfg, datamanager):
    models = dict()
    if cfg.model.adversarial is True:
        generator_S2R = torchreid.models.build_model(
            name=cfg.model.generator_name,
            adversarial=True
        )
        models[cfg.model.generator_name] = generator_S2R

        discriminator_R = torchreid.models.build_model(
            name=cfg.model.discriminator_name,
            adversarial=True
        )
        models[cfg.model.discriminator_name] = discriminator_R

        feature_net = torchreid.models.build_model(
            adversarial=True,
            name=cfg.model.feature_net_name,
            use_mlp=True,
            lr=cfg.train.feature_net_lr,
            weight_decay=cfg.train.feature_net_weight_decay,
            step_size=cfg.train.feature_net_stepsize,
            optim=cfg.train.feature_net_optim,
            nc=cfg.train.feature_net_nc,
            adam_beta1=cfg.adam.beta1
        )
        models[cfg.model.feature_net_name] = feature_net

        if cfg.loss.adversarial.weight_x > 0:
            convnet = torchreid.models.build_model(
                num_classes=datamanager.num_train_pids,
                in_planes=256,
                name=cfg.model.name,
                adversarial=True,
            )
            models[cfg.model.name] = convnet
    else:
        model = torchreid.models.build_model(
            name=cfg.model.name,
            num_classes=datamanager.num_train_pids,
            loss=cfg.loss.name,
            pretrained=cfg.model.pretrained,
            use_gpu=cfg.use_gpu,
            last_stride=cfg.model.last_stride,
        )
        models[cfg.model.name] = model
    return models


def make_optimizer(cfg, models):
    optimizers = dict()
    if cfg.model.adversarial is True:
        optimizer_GS2R = torchreid.optim.build_optimizer(
            cfg.model.generator_name,
            models[cfg.model.generator_name],
            **generator_optimizer_kwargs(cfg)
        )
        optimizers[cfg.model.generator_name] = optimizer_GS2R

        optimizer_DR = torchreid.optim.build_optimizer(
            cfg.model.discriminator_name,
            models[cfg.model.discriminator_name],
            **discriminator_optimizer_kwargs(cfg)
        )
        optimizers[cfg.model.discriminator_name] = optimizer_DR

        if cfg.loss.adversarial.weight_x > 0:
            optimizer_CNN = torchreid.optim.build_optimizer(
                cfg.model.name,
                models[cfg.model.name],
                **optimizer_kwargs(cfg)
            )
            optimizers[cfg.model.name] = optimizer_CNN
        optimizers[cfg.model.feature_net_name] = None  # Initialized on the fly when mlp receives the first batch
    else:
        optimizer = torchreid.optim.build_optimizer(
            cfg.model.name,
            models[cfg.model.name],
            **optimizer_kwargs(cfg)
        )
        optimizers[cfg.model.name] = optimizer
    return optimizers


def make_scheduler(cfg, optimizers):
    schedulers = dict()
    if cfg.model.adversarial is True:
        scheduler_GS2R = torchreid.optim.build_lr_scheduler(
            optimizers[cfg.model.generator_name],
            **generator_lr_scheduler_kwargs(cfg)
        )
        schedulers[cfg.model.generator_name] = scheduler_GS2R

        scheduler_DR = torchreid.optim.build_lr_scheduler(
            optimizers[cfg.model.discriminator_name],
            **discriminator_lr_scheduler_kwargs(cfg)
        )
        schedulers[cfg.model.discriminator_name] = scheduler_DR

        if cfg.loss.adversarial.weight_x > 0:
            scheduler_CNN = torchreid.optim.build_lr_scheduler(
                optimizers[cfg.model.name],
                **lr_scheduler_kwargs(cfg)
            )
            schedulers[cfg.model.name] = scheduler_CNN
        schedulers[cfg.model.feature_net_name] = None
    else:
        scheduler = torchreid.optim.build_lr_scheduler(
            optimizers[cfg.model.name],
            **lr_scheduler_kwargs(cfg)
        )
        schedulers[cfg.model.name] = scheduler
    return schedulers


def make_engine(cfg, datamanager, models, optimizers, schedulers):
    if cfg.model.adversarial is True:
        # if cfg.data.val is True:
        #     print("For domain adaptation there are no validation sets")
        #     raise ValueError
        if cfg.data.type == 'image':
            model_names = {
                'generator': cfg.model.generator_name,
                'discriminator': cfg.model.discriminator_name,
                'feature_net': cfg.model.feature_net_name,
            }
            if cfg.loss.adversarial.weight_x > 0:
                model_names['convnet'] = cfg.model.name

            engine = torchreid.engine.ImageAdversarialEngine(
                datamanager,
                model_names,
                models,
                optimizers=optimizers,
                schedulers=schedulers,
                use_gpu=cfg.use_gpu,
                val=cfg.data.val,
                epoch_id=cfg.train.epoch_id,
                weight_nce=cfg.loss.adversarial.weight_nce,
                weight_idt=cfg.loss.adversarial.weight_idt,
                weight_gen=cfg.loss.adversarial.weight_gen,
                weight_dis=cfg.loss.adversarial.weight_dis,
                weight_sim=cfg.loss.adversarial.weight_sim,
                weight_x=cfg.loss.adversarial.weight_x,
                weight_t=cfg.loss.adversarial.weight_t,
                sim_type_loss=cfg.loss.adversarial.sim_type_loss,
                guide_gen=cfg.loss.adversarial.guide_gen,
                nce_layers=cfg.loss.adversarial.nce_layers,
                dis_layers=cfg.loss.adversarial.dis_layers,
                num_patches=cfg.loss.adversarial.num_patches,
            )
        else:
            print("Adversairal training implemented only for image datasets")
            raise NotImplementedError
    else:
        if cfg.data.type == 'image':
            if cfg.loss.name == 'softmax':
                engine = torchreid.engine.ImageSoftmaxEngine(
                    datamanager,
                    cfg.model.name,
                    models[cfg.model.name],
                    optimizer=optimizers[cfg.model.name],
                    scheduler=schedulers[cfg.model.name],
                    use_gpu=cfg.use_gpu,
                    label_smooth=cfg.loss.softmax.label_smooth,
                    val=cfg.data.val,
                )

            else:
                engine = torchreid.engine.ImageTripletEngine(
                    datamanager,
                    cfg.model.name,
                    models[cfg.model.name],
                    optimizer=optimizers[cfg.model.name],
                    margin=cfg.loss.triplet.margin,
                    weight_t=cfg.loss.triplet.weight_t,
                    weight_x=cfg.loss.triplet.weight_x,
                    scheduler=schedulers[cfg.model.name],
                    use_gpu=cfg.use_gpu,
                    label_smooth=cfg.loss.softmax.label_smooth,
                    val=cfg.data.val,
                    generator_path=cfg.train.generator_path
                )

        else:
            if cfg.loss.name == 'softmax':
                engine = torchreid.engine.VideoSoftmaxEngine(
                    datamanager,
                    models[cfg.model.name],
                    optimizer=optimizers['optimizer'],
                    scheduler=schedulers['scheduler'],
                    use_gpu=cfg.use_gpu,
                    label_smooth=cfg.loss.softmax.label_smooth,
                    pooling_method=cfg.video.pooling_method,
                )

            else:
                engine = torchreid.engine.VideoTripletEngine(
                    datamanager,
                    models[cfg.model.name],
                    optimizer=optimizers['optimizer'],
                    margin=cfg.loss.triplet.margin,
                    weight_t=cfg.loss.triplet.weight_t,
                    weight_x=cfg.loss.triplet.weight_x,
                    scheduler=schedulers['scheduler'],
                    use_gpu=cfg.use_gpu,
                    label_smooth=cfg.loss.softmax.label_smooth,
                )

    return engine


def reset_config(cfg, args):
    if args.root:
        cfg.data.root = args.root
    if args.sources:
        cfg.data.sources = args.sources
    if args.targets:
        cfg.data.targets = args.targets
    if args.transforms:
        cfg.data.transforms = args.transforms


def check_cfg(cfg):
    if cfg.loss.name == 'triplet' and cfg.loss.triplet.weight_x == 0:
        assert cfg.train.fixbase_epoch == 0, \
            'The output of classifier is not included in the computational graph'


def make_wandb_config(cfg):
    config = {}

    # Data
    config['sources'] = cfg.data.sources
    config['targets'] = cfg.data.targets
    config['combineall'] = cfg.data.combineall
    config['transforms'] = cfg.data.transforms
    config['load_train_targets'] = cfg.data.load_train_targets
    config['val'] = cfg.data.val
    if 'gta_synthreid' in cfg.data.sources:
        config['gta_samples_per_identity'] = cfg.data.n_samples

    # Train
    config['lr'] = cfg.train.lr
    config['optim'] = cfg.train.optim
    config['weight_decay'] = cfg.train.weight_decay
    config['lr_scheduler'] = cfg.train.lr_scheduler
    config['stepsize'] = cfg.train.stepsize
    config['gamma'] = cfg.train.gamma
    config['epochs'] = cfg.train.max_epoch
    config['batch_size'] = cfg.train.batch_size
    config['seed'] = cfg.train.seed
    if cfg.train.lr_scheduler == 'warmup_multi_step':
        config['warmup_iters'] = cfg.train.warmup_iters
    if cfg.train.fixbase_epoch > 0:
        config['fixbase_epoch'] = cfg.train.fixbase_epoch
        config['open_layers'] = cfg.train.open_layers
    if cfg.train.staged_lr is True:
        config['staged_lr'] = cfg.train.staged_lr
        config['new_layers'] = cfg.train.new_layers
    if cfg.model.adversarial is True:
        config['generator_lr'] = cfg.train.generator_lr
        config['discriminator_lr'] = cfg.train.discriminator_lr
        config['feature_net_lr'] = cfg.train.feature_net_lr
        config['generator_optim'] = cfg.train.generator_optim
        config['discriminator_optim'] = cfg.train.discriminator_optim
        config['feature_net_optim'] = cfg.train.feature_net_optim
        config['generator_weight_decay'] = cfg.train.generator_weight_decay
        config['discriminator_weight_decay'] = cfg.train.discriminator_weight_decay
        config['feature_net_weight_decay'] = cfg.train.feature_net_weight_decay
        config['generator_lr_scheduler'] = cfg.train.generator_lr_scheduler
        config['discriminator_lr_scheduler'] = cfg.train.discriminator_lr_scheduler
        config['generator_stepsize'] = cfg.train.generator_stepsize
        config['discriminator_stepsize'] = cfg.train.discriminator_stepsize
        config['feature_net_stepsize'] = cfg.train.feature_net_stepsize
        config['generator_gamma'] = cfg.train.generator_gamma
        config['discriminator_gamma'] = cfg.train.discriminator_gamma
        config['feature_net_nc'] = cfg.train.feature_net_nc
        config['weight_nce'] = cfg.loss.adversarial.weight_nce
        config['weight_idt'] = cfg.loss.adversarial.weight_idt
        config['weight_gen'] = cfg.loss.adversarial.weight_gen
        config['weight_dis'] = cfg.loss.adversarial.weight_dis
        config['weight_sim'] = cfg.loss.adversarial.weight_sim
        config['weight_x'] = cfg.loss.adversarial.weight_x
        config['sim_type_loss'] = cfg.loss.adversarial.sim_type_loss
        config['guide_gen'] = cfg.loss.adversarial.guide_gen
        config['nce_layers'] = cfg.loss.adversarial.nce_layers
        config['dis_layers'] = cfg.loss.adversarial.dis_layers
        config['num_patches'] = cfg.loss.adversarial.num_patches

    # Optimizer
    if cfg.train.optim == 'adam':
        config['beta1'] = cfg.adam.beta1
        config['beta2'] = cfg.adam.beta2
    elif cfg.train.optim == 'sgd':
        config['momentum'] = cfg.sgd.momentum

    # Loss
    if not cfg.model.adversarial:
        config['loss'] = cfg.loss.name
        config['label_smooth'] = cfg.loss.softmax.label_smooth
        if cfg.loss.name == 'triplet':
            config['triplet_margin'] = cfg.loss.triplet.margin
            config['triplet_weight_t'] = cfg.loss.triplet.weight_t
            config['triplet_weight_x'] = cfg.loss.triplet.weight_x
    else:
        pass

    # Test
    config['dist_metric'] = cfg.test.dist_metric

    return config


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config-file', type=str, default='', help='path to config file'
    )
    parser.add_argument(
        '--run_id', type=str, default='', help='wandb run id to resume the training or to test'
    )
    parser.add_argument(
        '--mode', type=str, default='', help='wandb mode for disabling logs'
    )
    parser.add_argument(
        '-s',
        '--sources',
        type=str,
        nargs='+',
        help='source datasets (delimited by space)'
    )
    parser.add_argument(
        '-t',
        '--targets',
        type=str,
        nargs='+',
        help='target datasets (delimited by space)'
    )
    parser.add_argument(
        '--transforms', type=str, nargs='+', help='data augmentation'
    )
    parser.add_argument(
        '--root', type=str, default='', help='path to data root'
    )
    parser.add_argument(
        'opts',
        default=None,
        nargs=argparse.REMAINDER,
        help='Modify config options using the command-line'
    )
    args = parser.parse_args()

    cfg = get_default_config()
    cfg.use_gpu = torch.cuda.is_available()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    reset_config(cfg, args)
    cfg.merge_from_list(args.opts)
    set_random_seed(cfg.train.seed)
    check_cfg(cfg)

    run_id = args.run_id
    mode = args.mode
    run_name = args.config_file.split("/")[-1].split(".")[0]
    log_name = 'test.log' if cfg.test.evaluate else 'train.log'
    log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
    sys.stdout = Logger(osp.join(cfg.data.save_dir, log_name))

    print('Show configuration\n{}\n'.format(cfg))
    print('Collecting env info ...')
    print('** System info **\n{}\n'.format(collect_env_info()))

    # if cfg.use_gpu:
    #     torch.backends.cudnn.benchmark = True

    datamanager = make_datamanager(cfg)
    print("Training classes: ", datamanager.num_train_pids)
    print('Building model(s): {}'.format(cfg.model.name))
    models = make_model(cfg, datamanager)
    # print(model)

    for name in models.keys():
        if name == 'mlp':
            print('The feature_net is created on the fly when it sees it first bath of data')
        elif name == 'id_net':
            num_params, flops = compute_model_complexity(
                models[name], (1, 256, 64, 32)
            )
            print('{} complexity: params={:,} flops={:,}'.format(name, num_params, flops))

        else:
            num_params, flops = compute_model_complexity(
                models[name], (1, 3, cfg.data.height, cfg.data.width)
            )
            print('{} complexity: params={:,} flops={:,}'.format(name, num_params, flops))

        if cfg.model.load_weights and check_isfile(cfg.model.load_weights):
            if cfg.model.adversarial is True:
                path = cfg.model.load_weights.split('generator')[0] + name +\
                    cfg.model.load_weights.split('generator')[1]
            else:
                path = cfg.model.load_weights
            load_pretrained_weights(models[name], path)
        if cfg.use_gpu:
            models[name] = nn.DataParallel(models[name]).cuda()
    # sys.exit()
    optimizers = make_optimizer(cfg, models)
    schedulers = make_scheduler(cfg, optimizers)

    if cfg.model.resume and check_isfile(cfg.model.resume):
        # if cfg.model.adversarial is True:
        for name in models.keys():
            if name == 'mlp':
                dummy_feats = [torch.rand((1, 3)), torch.rand((1, 128))]
                dummy_feats.extend([torch.rand((1, 256))] * 3)
                models[name].module.create_mlp(dummy_feats)
                optimizer_mlp, scheduler_mlp = models[name].module.optim_sched()
                optimizers['mlp'] = optimizer_mlp
                schedulers['mlp'] = scheduler_mlp

            if cfg.model.adversarial is True:
                path = cfg.model.resume.split('generator')[0] + name + cfg.model.resume.split('generator')[1]
            else:
                path = cfg.model.resume

            cfg.train.start_epoch = resume_from_checkpoint(
                path, models[name], optimizer=optimizers[name], scheduler=schedulers[name]
            )
    if cfg.model.adversarial is True:
        file_name = 'wandb_adv_api.key'
    else:
        file_name = 'wandb_base_api.key'
    try:
        with open(file_name, 'r') as file:
            wandb_identity = json.load(file)
    except FileNotFoundError as e:
        print("Create a json file in the root directory of this repo called 'wandb_adv_api.key'\
                or 'wandb_base_api.key',\n")
        print("it must have as keys the 'entity', 'project' and 'api-key' of your wandb account.\n")
        print("For the entity you can put yours or that of a team,\
                for the project put a project name (e.g. adversarial_ReId")
        raise(e)
    wandb.login(anonymous='never', relogin=True, timeout=30, key=wandb_identity['key'])
    relevant_hyperparams = make_wandb_config(cfg)
    # wandb.config = relevant_hyperparams
    mode = "disabled" if cfg.test.evaluate is True else mode
    wandb.init(resume=True if (cfg.model.resume != '' or cfg.model.load_weights != '') else False,
               # sync_tensorboard=True,
               mode=mode if mode != "" else None,
               id=run_id if run_id != "" else None,
               config=relevant_hyperparams,
               project=wandb_identity['project'],
               entity=wandb_identity['entity'],
               name=run_name)
    # wandb.watch(models['id_net'].module, log='all', log_freq=1)
    # last_step = run.history._step

    if cfg.model.adversarial is True:
        print(
            'Building adversarial-engine for {}-reid'.format(cfg.data.type)
        )

    else:
        print(
            'Building {}-engine for {}-reid'.format(cfg.loss.name, cfg.data.type)
        )
    engine = make_engine(cfg, datamanager, models, optimizers, schedulers)
    engine.run(**engine_run_kwargs(cfg))

    # if not cfg.model.adversarial:
    #     engine.run(
    #         test_only=True,
    #         **engine_run_kwargs(cfg)
    #     )


if __name__ == '__main__':
    main()
