from torch.optim import lr_scheduler
import logging


class Custom_Scheduler:
    def __init_(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def warmup_cos_annealing(self, optimizer, **kwargs):
        warmup_epochs = kwargs.get('warmup_epochs', 8)
        total_epochs = kwargs.get('total_epochs', 100)
        # ウォームアップ時の学習率スケール関数
        def warmup_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch) / warmup_epochs  # 線形増加
            return 1.0  # その後は 1.0（CosineAnnealingLR へ引き継ぐ）

        warmup_scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)

        # CosineAnnealing の T_max 設定
        cosine_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=0)

        # SequentialLR でスケジューラを切り替え
        scheduler = lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )

        return scheduler