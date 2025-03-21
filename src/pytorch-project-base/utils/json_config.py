def config_json_element_check(config):
    # Check if the config file is in the correct format
    assert "seed" in config, "seed is not in config"
    assert "log_config" in config, "log_config is not in config"

    assert "dataloader" in config, "dataloader is not in config"
    assert "type" in config["dataloader"], "type is not in dataloader"

    assert "optimizer" in config, "optimizer is not in config"
    assert "type" in config["optimizer"], "type is not in optimizer"

    assert "criterion" in config, "criterion is not in config"
    assert "type" in config["criterion"], "type is not in criterion"

    assert "model" in config, "model is not in config"
    assert "type" in config["model"], "type is not in model"
    assert "torchvision_model" in config["model"], "torchvision_model is not in model"
    assert "timm_model" in config["model"], "timm_model is not in model"
    assert "pretrained" in config["model"], "pretrained is not in model"

    assert "trainer" in config, "trainer is not in config"
    assert "epochs" in config["trainer"], "epochs is not in trainer"
    assert "save_period" in config["trainer"], "save_period is not in trainer"
    assert "dataparallel" in config["trainer"], "dataparallel is not in trainer"
    assert "huggingface" in config["trainer"], "huggingface is not in trainer"