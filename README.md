# docker image for execute google-research vision_transformer.

https://github.com/google-research/vision_transformer

# to build tree step

1. build jax-lib whl file
``` terminal
$ docker build -f Dockerfile-build -t jax-build .
$ docker run --rm -it -v /tmp:/work jax-build /bin/bash
```

in docker contener.

``` terminal
# cd /work
# git clone https://github.com/google/jax
# cd jax
# python build/build.py --enable_cuda
# exit
```

``` terminal
$ cp /tmp/jax/dist/*.whl .
```

2. build vision_transformer docker image
```
$ docker build -t vision_transformer .
```

# to execute visin transformer

``` terminal
$ docker run --rm vision_transformer --help
usage: train.py [-h] --name NAME
                [--model {ViT-B_16,ViT-B_32,ViT-L_16,ViT-L_32,ViT-H_14,testing}]
                --logdir LOGDIR [--vit_pretrained_dir VIT_PRETRAINED_DIR]
                [--output OUTPUT] [--copy_to COPY_TO] --dataset
                {cifar10,cifar100,imagenet2012}
                [--tfds_manual_dir TFDS_MANUAL_DIR]
                [--tfds_data_dir TFDS_DATA_DIR] [--mixup_alpha MIXUP_ALPHA]
                [--grad_norm_clip GRAD_NORM_CLIP]
                [--optim_dtype {bfloat16,float32}] [--total_steps TOTAL_STEPS]
                [--accum_steps ACCUM_STEPS] [--batch BATCH]
                [--batch_eval BATCH_EVAL] [--shuffle_buffer SHUFFLE_BUFFER]
                [--prefetch PREFETCH] [--base_lr BASE_LR]
                [--decay_type {cosine,linear}] [--warmup_steps WARMUP_STEPS]
                [--eval_every EVAL_EVERY] [--progress_every PROGRESS_EVERY]

Fine-tune ViT-M model.

optional arguments:
  -h, --help            show this help message and exit
  --name NAME           Name of this run. Used for monitoring and
                        checkpointing.
  --model {ViT-B_16,ViT-B_32,ViT-L_16,ViT-L_32,ViT-H_14,testing}
                        Which variant to use; ViT-M gives best results.
  --logdir LOGDIR       Where to log training info (small).
  --vit_pretrained_dir VIT_PRETRAINED_DIR
                        Where to search for pretrained ViT models.
  --output OUTPUT       Where to store the fine tuned model checkpoint.
  --copy_to COPY_TO     Directory where --logdir and --output should be
                        stored. This directory can be on any filesystem
                        accessible through by tf.io.gfile
  --dataset {cifar10,cifar100,imagenet2012}
                        Choose the dataset. It should be easy to add your own!
                        Do not forget to set --tfds_manual_dir if necessary.
  --tfds_manual_dir TFDS_MANUAL_DIR
                        Path to manually downloaded dataset.
  --tfds_data_dir TFDS_DATA_DIR
                        Path to tensorflow_datasets directory.
  --mixup_alpha MIXUP_ALPHA
                        Coefficient for mixup combination. See
                        https://arxiv.org/abs/1710.09412
  --grad_norm_clip GRAD_NORM_CLIP
                        Resizes global gradients.
  --optim_dtype {bfloat16,float32}
                        Datatype to use for momentum state.
  --total_steps TOTAL_STEPS
                        Number of steps; determined by hyper module if not
                        specified.
  --accum_steps ACCUM_STEPS
                        Accumulate gradients over multiple steps to save on
                        memory.
  --batch BATCH         Batch size for training.
  --batch_eval BATCH_EVAL
                        Batch size for evaluation.
  --shuffle_buffer SHUFFLE_BUFFER
                        Shuffle buffer size.
  --prefetch PREFETCH   Number of batches to prefetch to device.
  --base_lr BASE_LR     Base learning-rate for fine-tuning. Most likely
                        default is best.
  --decay_type {cosine,linear}
                        How to decay the learning rate.
  --warmup_steps WARMUP_STEPS
                        How to decay the learning rate.
  --eval_every EVAL_EVERY
                        Run prediction on validation set every so many
                        steps.Will always run one evaluation at the end of
                        training.
  --progress_every PROGRESS_EVERY
                        Log progress every so many steps.
$ docker run --name vision_transformer --gpus all vision_transformer --name ViT-B_16-cifar10_`date +%F_%H%M%S` --model ViT-B_16 --logdir /tmp/vit_logs  --dataset cifar10
```
