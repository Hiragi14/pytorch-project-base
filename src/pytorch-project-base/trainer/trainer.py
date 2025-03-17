import torch
from base import BaseTrainer


class Trainer(BaseTrainer):
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