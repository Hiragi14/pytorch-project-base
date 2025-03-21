# pytorch-project-base
PyTorchによる深層モデル(Computer Vision)の学習を簡単にするプロジェクト．
Wandbによるロギング機能をサポート．

## Index

- [pytorch-project-base](#pytorch-project-base)
  - [Index](#index)
  - [Usage](#usage)
  - [カスタムローダーの使用](#カスタムローダーの使用)
  - [Floder](#floder)

## Usage
- you have to change dataset dir on demand in your `config.json`
- データセット保存パスを環境に応じて変更してください．
```json:config.json
{
    "name": "mnist_mlp",

    "dataloader": {
        "type": "MnistDataLoader",
        "args":{
            "data_dir": "/DeepLearning/Dataset/torchvision/MNIST", //change dataset dir on demand
            "batch_size": 128,
            "shuffle": true,
            "num_workers": 2,
            "validation_split": 0.0,
            "download": false
        }
    },

    "optimizer": {
        "type": "SGD",
        "args": {
            "lr": 0.01
        }
    },

    "criterion": {
        "type": "CrossEntropyLoss",
        "args": {}
    },

    "scheduler": {
        "type": "StepLR",
        "args": {
            "step_size": 7,
            "gamma": 0.1
        }
    },

    "model": {
        "torchvision_model": false,
        "timm_model": false,
        "pretrained": false,
        "type": "MLP",
        "args": {
            "layer_sizes": [784, 128, 64, 100, 10]
        }
    },

    "trainer": {
        "epochs": 1,
        "save_period": 2,
        "checkpoint_dir": "saved_models/", // change model save dir on demand
        "huggingface": {
            "use": true,
            "token": "hogehoge",
            "repo": "base/project"
        }
    },

    "web_logger": { # this is optional
        "type": "WandbLogger",
        "args": {
            "project": "timm_model_test",
            "name": "deit_tiny_patch16_224",
            "tags": ["deit_tiny_patch16_224", "ImageNet1k"]
        }
    }
}
```
- you can start training by this command
- `config.json` support an MNIST example on MLP
- 以下のコマンドでMLPのMNISTタスクでの学習が始まります．
```bash
python train.py -c config.json
# python train.py --config config.json
# python train.py
```

## カスタムローダーの使用
- pythonモジュールとしてローダーを用意して読み込むか，src/pytorch-project-base/ 直下に読み込める形で配置してください．
```json
    "dataloader": {
        "type": "CustomLoadImageNetDataLoader",
        "args":{
            "data_dir": "/workspace/ImageNet",
            "batch_size": 128,
            "shuffle": true,
            "num_workers": 4,
            "validation_split": 0.0,
            "download": false
        },
        "loader":{
            "module": "pytorch_dct_loader",
            "function": "return_loader_normalize",
            "args": {
                "size": 224,
                "block_size": 16,
                "alpha": 1.0
            }
        }
    },
```


- you can test functions by pytest
- 関数のテストは以下のように行っています．
```bash
$ pytest test
```

## Floder
```
pytorch-project-base/
│
├── train.py    # script to start training
│
├── config.json # holds configuration for training
│
├── src/
│   ├── base/ # abstract base classes
│   │   ├── base_dataloader.py
│   │   ├── base_model.py
│   │   └── base_trainer.py
│   │
│   ├── dataloader/ # data loading
│   │   └── dataloaders.py
│   │
│   ├── model/ # models
│   │   ├── mlp.py
│   │   ├── movileNetV1.py
│   │   └── smallVGG.py
│   │
│   ├── trainer/ # trainers
│   │   └── trainer.py
│   │  
│   └── utils/ # utility functions
│       ├── criterion.py
│       ├── scheduler.py
│       └── optimizer.py
│
├── test/
│   ├── unit_test/
│   │   ├── criterion_test.py
│   │   ├── scheduler_test.py
│   │   ├── dataloader_test.py
│   │   └── optimizer_test.py
│   │
│   ├──
│   ├──
│   └──
│
│

```