from sacred import Experiment
ex = Experiment("my_clip")

def _loss_names(d):
    # 返回本次任务
    ret = {
        "itm": 0,
        "mlm": 0,
        "irtr": 0,
    }
    ret.update(d)
    return ret

# 默认配置中必须包含所有参数
# 即named_config中不能出现config里没有的参数
@ex.config
def default_config():
    # 常规设置
    # below params varies with the environment
    data_root = "/home/tzj/datas/mscoco/coco2014_karpathy_prepared"
    output_dir = "../outputs"
    log_dir = "../logs"
    exp_name = 'my_clip'
    seed = 0
    datasets = ['coco']
    train_dataset_len = -1
    val_dataset_len = -1
    test_dataset_len = -1
    shuffle = True # 训练集是否打乱
    loss_name = _loss_names({'itm': 1, 'mlm': 1})

    # Image Setting
    vit_name = ''
    vit = ''
    train_transform_keys = ['clip']
    val_transform_keys = ['clip']
    image_size = 224
    patch_size = 32
    # draw_false_image = 1
    image_only = False

    # Text Setting
    max_text_len = 40
    tokenizer_name = ''
    tokenizer = ''
    vocab_size = 0
    whole_word_masking = False # note that whole_word_masking does not work for RoBERTa
    mlm_prob = 0.15 # mlm遮罩比例
    # draw_false_text = 0

    # Transformer Setting
    num_top_layer = 6  # 融合模块的层数
    input_image_embed_size = 768
    input_text_embed_size = 768
    hidden_size = 768
    num_heads = 12 # 注意力的head数量
    num_layers = 6 #
    # mlp_ratio = 1 # 中间层的维度：hidden_size * mlp_ratio
    mlp_ratio = 4 # 中间层的维度：hidden_size * mlp_ratio
    drop_rate = 0.1 # dropout

    # Optimizer Setting
    optim_type = ''
    learning_rate = 2e-5
    eps = 0.0
    betas = (0.0, 0.0)
    momentum = 0.0
    weight_decay = 0.01
    decay_power = 1
    max_epoch = 100
    max_steps = 100000
    warmup_steps = 10000
    end_lr = 0
    lr_mult_head = 5  # multiply lr for downstream heads
    lr_mult_cross_modal = 5  # multiply lr for the cross-modal module

    # Downstream Setting
    get_recall_metric = False

    # Lightning Trainer Setting
    num_sanity_val_steps = 0 # 在开始前取 n 个val batches
    fast_dev_run = False # 快速检验，取 n 个train, val, test batches
    val_check_interval = 0.5 # 验证间隔
    test_only = False

    batch_size = 128  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.
    per_gpu_batchsize = 32  # you should define this manually with per_gpu_batch_size=#

    queue_size = 57600
    momentum = 0.995

    gpus = [0]
    num_nodes = 1
    pin_memory = True
    load_path = ""  # 模型权重地址
    # load_path = "/home/tzj/pretrained_models/meter_clip16_224_roberta_pretrain.ckpt"  # 模型权重地址
    num_workers = 8
    precision = 32
    max_grad_norm = 1.
    apex = False
    amp_level = 'O1'

@ex.named_config
def adamw_optim():
    optim_type = 'adamw'
    learning_rate = 1e-5
    eps = 1e-8
    betas = (0.9, 0.98)
    weight_decay = 0.01
    max_epoch = 100
    max_steps = 100000
    warmup_steps = 10000
    end_lr = 0
    lr_mult_head = 5 # multiply lr for downstream heads
    lr_mult_cross_modal = 5  # multiply lr for the cross-modal module

@ex.named_config
def task_finetune_irtr_coco_clip_bert():
    exp_name = 'finetune_irtr_coco'
    datasets = ['coco']
    shuffle = True  # 训练集是否打乱
    loss_name = _loss_names({'itm':0, 'irtr':1})
    # batch_size = 128
    max_epoch = 20
    max_steps = -1
    warmup_steps = 0.2
    get_recall_metric = True

    # 调整优化器参数
    learning_rate = 2e-5
    # learning_rate = 5e-6
    lr_mult_head= 5
    lr_mult_cross_modal = 5
    image_size = 288

# vision encoder
@ex.named_config
def clip32():
    vit_name = 'ViT-B-32'
    vit = '/home/tzj/pretrained_models/ViT/ViT-B-32.pt'  # vit模型权重
    image_size = 224 # 调整后的图片像素
    patch_size = 32 # 送入vit的图片块的像素
    train_transform_keys = ["clip"] # 预处理图片的模型（将图片处理为image_size * image_size）
    val_transform_keys = ["clip"]
    input_image_embed_size = 768

@ex.named_config
def clip16():
    vit_name = 'ViT-B-16'
    vit = '/home/tzj/pretrained_models/ViT/ViT-B-16.pt'  # vit模型权重
    image_size = 224  # 调整后的图片像素
    patch_size = 16  # 送入vit的图片块的像素
    train_transform_keys = ["clip"]  # 预处理图片的模型（将图片处理为image_size * image_size）
    val_transform_keys = ["clip"]
    input_image_embed_size = 768

# text encoder
@ex.named_config
def text_roberta():
    tokenizer_name = 'roberta-base'
    tokenizer = '/home/tzj/pretrained_models/en-roberta-base'
    vocab_size = 50265
    input_text_embed_size = 768


@ex.named_config
def text_roberta_large():
    tokenizer_name = 'roberta-large'
    tokenizer = 'roberta-large'
    vocab_size = 50265
    input_text_embed_size = 1024


# random augmentatioin
@ex.named_config
def imagnet_randaug():
    train_transform_keys = ['imagenet_randaug']

@ex.named_config
def clip_randaug():
    train_transform_keys = ['clip_randaug']

@ex.named_config
def debug():
    seed = 0
    train_dataset_len = 500
    val_dataset_len = 200
    test_dataset_len = 50
    fast_dev_run = 2
    shuffle = False
    num_workers = 0


