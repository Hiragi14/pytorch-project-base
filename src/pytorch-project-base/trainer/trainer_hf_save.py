import torch
import os
from datetime import datetime
from base import BaseTrainer
from huggingface_hub import HfApi, create_repo


class Trainer_HF(BaseTrainer):
    def __init__(self, model, criterion, optimizer, config, device, 
                 loader_train, loader_valid, scheduler=None):
        super().__init__(model, criterion, optimizer, config, device, loader_train, loader_valid)

        self.scheduler = scheduler

    def _train_epoch(self, loader, pbar):
        self.model.train()

        total = 0
        correct = 0
        cumulative_loss = 0

        for img, lbl in loader:
            self.optimizer.zero_grad()
            
            img = img.to(self.device)
            lbl = lbl.to(self.device)
            out = self.model.forward(img)
            
            loss = self.criterion(out, lbl)
            loss.backward()
            self.optimizer.step()
            
            # 精度と損失を更新
            correct += (out.argmax(dim=1) == lbl).sum().item()
            total += out.shape[0]
            cumulative_loss += loss.item()

            # tqdmのプログレスバーの後ろにカスタムメッセージを表示
            pbar.set_postfix(loss=loss.item(), accuracy=(correct / total) * 100)
            pbar.update(1)
        
        self.scheduler.step()
        
        return correct / total, cumulative_loss / len(loader)
    
    def _valid_epoch(self, loader, pbar):
        self.model.eval()
    
        total = 0
        correct = 0
        cumulative_loss = 0
            
        for img, lbl in loader:
            img = img.to(self.device)
            lbl = lbl.to(self.device)
            with torch.no_grad():
                out = self.model(img)
                loss = self.criterion(out, lbl)
                
                # 精度と損失を更新
                correct += (out.argmax(dim=1) == lbl).sum().item()
                total += out.shape[0]
                cumulative_loss += loss.item()
                
                # tqdmのプログレスバーの後ろにカスタムメッセージを表示
                pbar.set_postfix(loss=loss.item(), accuracy=(correct / total) * 100)
                pbar.update(1)
        
        return correct / total, cumulative_loss / len(loader)
    
    
    def _save_checkpoint(self, epoch, result):
        super()._save_checkpoint(epoch, result)
        if epoch % self.save_period == 0:
            self._huggingface_model_save()
    
    
    def _save_model(self):
        super()._save_model()
        self._huggingface_model_save()
        
    
    def _huggingface_model_save(self):
        try:
            repo = self.config["trainer"]["huggingface"]["repo"]
        except:
            self.logger.warning("huggingface model saving used, but huggingface token or repo is empty.")
            return
        
        api = HfApi()
        
        try:
            current_date = datetime.now().strftime("%Y-%m-%d")
            # 重みファイルのアップロード
            api.upload_file(
                path_or_fileobj=self.save_dir,
                path_in_repo=f"{current_date}/{self.save_dir}",
                repo_id=repo,
            )
            
            print(f"重みファイルが正常にアップロードされました: https://huggingface.co/{repo}")
            self.logger.info(f"重みファイルが正常にアップロードされました: https://huggingface.co/{repo}")
        except Exception as ex:
            print(f"重みファイルのアップロードに失敗しました: https://huggingface.co/{repo}")
            self.logger.warning(f"FAILED to save model on huggingface! Exception: {ex}")
        