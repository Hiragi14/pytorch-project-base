# モデル登録
MODEL_REGISTRY = {}

# モデル登録用デコレータ
def register_model(name):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator