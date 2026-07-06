import os
import torch
from tqdm import tqdm
import sklearn
import utils
import numpy as np
from src.dataload import ASDDataset
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.net = kwargs['net']
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = self.args.writer
        self.logger = self.args.logger
        #self.criterion = ASDLoss().to(self.args.device)
        self.transform = kwargs['transform']



    def train(self, train_loader):

        self.net.eval()
        with torch.no_grad():
            self.net.set_c(train_loader) # 设定超球面中心

        # self.test(save=False)
        model_dir = os.path.join(self.writer.log_dir, 'model')
        os.makedirs(model_dir, exist_ok=True)
        epochs = self.args.epochs
        valid_every_epochs = self.args.valid_every_epochs
        early_stop_epochs = self.args.early_stop_epochs
        start_valid_epoch = self.args.start_valid_epoch
        num_steps = len(train_loader)
        self.sum_train_steps = 0
        self.sum_valid_steps = 0
        best_metric = 0
        no_better_epoch = 0

        for epoch in range(0, epochs + 1):
            
            
            # train
            sum_total_loss = 0
            sum_svdd_loss = 0
            sum_classification_loss = 0
            sum_contrastive_loss = 0
            self.net.train()
            train_bar = tqdm(train_loader, total=num_steps, desc=f'Epoch-{epoch}')

            # 混合训练
            for (x_wavs, x_mels, labels) in train_bar:
                # forward
                x_wavs, x_mels = x_wavs.float().to(self.args.device), x_mels.float().to(self.args.device)
                labels = labels.reshape(-1).long().to(self.args.device)
                x_mels = x_mels.unsqueeze(1)
                x_wavs = x_wavs.unsqueeze(1)

                # svdd_loss = self.net.compute_oneclass_loss(x_mels, labels)  # 单分类超球面损失
                # svdd_loss = self.net.compute_loss(x_mels,x_wavs,labels) # 多分类超球面损失， 根据对应的标签计算对应的超球面损失

                svdd_loss,_ = self.net.compute_soft_svdd_loss(x_mels,x_wavs,labels) # 软边界多分类超球面损失


                loss = svdd_loss #+ classfication_loss #+ loss_sup #

                train_bar.set_postfix(loss=f'{loss.item():.5f}')
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                
                # self.scheduler.step()
                
                

                # visualization
                self.writer.add_scalar(f'total_loss', loss.item(), self.sum_train_steps)
                self.writer.add_scalar(f'svdd_loss', svdd_loss.item(), self.sum_train_steps)


                sum_total_loss += loss.item()
                sum_svdd_loss += svdd_loss.item()
                #sum_classification_loss += classfication_loss.item()
                #sum_contrastive_loss += loss_sup.item()
                self.sum_train_steps += 1

            # current_lr = self.optimizer.param_groups[0]["lr"]
            # print(f"Epoch [{epoch+1}/{epochs}] "
            #         f"lr={current_lr:.6e} ")

            avg_total_loss = sum_total_loss / num_steps
            avg_svdd_loss = sum_svdd_loss / num_steps
            avg_classification_loss = sum_classification_loss / num_steps
            avg_contrastive_loss = sum_contrastive_loss / num_steps

            if self.scheduler is not None and epoch >= self.args.start_scheduler_epoch:
                self.scheduler.step()
            self.logger.info(f'Epoch-{epoch}\ttotal_loss:{avg_total_loss:.5f}, '
                             f'svdd_loss:{avg_svdd_loss:.5f}, '
                             f'classification_loss:{avg_classification_loss:.5f},'
                             f'contrastive_loss:{avg_contrastive_loss:.5f}')


            # valid
            if (epoch - start_valid_epoch) % valid_every_epochs == 0 and epoch >= start_valid_epoch:
                avg_auc, avg_pauc = self.valid(save=False)
                self.writer.add_scalar(f'auc', avg_auc, epoch)
                self.writer.add_scalar(f'pauc', avg_pauc, epoch)
                if avg_auc + avg_pauc >= best_metric:
                    no_better_epoch = 0
                    best_metric = avg_auc + avg_pauc
                    best_model_path = os.path.join(model_dir, 'best_checkpoint.pth.tar')
                    utils.save_model_state_dict(best_model_path, epoch=epoch,
                                                net=self.net.module if self.args.dp else self.net,
                                                optimizer=None)
                    self.logger.info(f'Best epoch now is: {epoch:4d}')
                else:
                    # early stop
                    no_better_epoch += 1
                    if no_better_epoch > early_stop_epochs > 0: break

            # save last 10 epoch state dict
            if epoch >= self.args.start_save_model_epochs:
                if (epoch - self.args.start_save_model_epochs) % self.args.save_model_interval_epochs == 0:
                    model_path = os.path.join(model_dir, f'{epoch}_checkpoint.pth.tar')
                    utils.save_model_state_dict(model_path, epoch=epoch,
                                                net=self.net.module if self.args.dp else self.net,
                                                optimizer=None)


    def valid(self, save=False):

        sum_auc, sum_pauc, num = 0, 0, 0
        result_dir = os.path.join(self.args.result_dir, self.args.version)

        
        self.net.eval()
        net = self.net.module if self.args.dp else self.net
        print('\n' + '=' * 20)

        for index, (target_dir, train_dir) in enumerate(
                zip(sorted(self.args.valid_dirs), sorted(self.args.train_dirs))):
            machine_type = target_dir.split('/')[-2]
            # result csv

            performance = []
            # get machine list
            machine_id_list = utils.get_machine_id_list(target_dir)
            for id_str in machine_id_list:
                test_files, y_true = utils.create_test_file_list(target_dir, id_str, dir_name='test')
                y_pred = []
                valid_dataset = ASDDataset(self.args, test_files, load_in_memory=False)
                valid_dataloader = DataLoader(valid_dataset, batch_size=self.args.batch_size,
                                               shuffle=False, num_workers=self.args.num_workers)

                for i,(x_wav, x_mel, label) in enumerate(valid_dataloader):
                    x_wav, x_mel = x_wav.float().to(self.args.device), x_mel.float().to(self.args.device)
                    x_mel = x_mel.unsqueeze(1)
                    x_wav = x_wav.unsqueeze(1)

                    label = label.long().to(self.args.device)
                    with torch.no_grad():
                        svdd_anomaly_score = net.compute_anomaly_score(x_mel, x_wav, label)
                    svdd_anomaly_score = svdd_anomaly_score.squeeze().cpu().numpy()
                    

                    y_pred.extend(svdd_anomaly_score)

                # compute auc and pAuc
                max_fpr = 0.1
                auc = sklearn.metrics.roc_auc_score(y_true, y_pred)
                p_auc = sklearn.metrics.roc_auc_score(y_true, y_pred, max_fpr=max_fpr)

                # 如果auc是NaN，跳过不累加
                if np.isnan(auc) or np.isnan(p_auc):
                    self.logger.warning(f'Skipping NaN AUC: auc={auc}, p_auc={p_auc}')
                    continue

                performance.append([auc, p_auc])

            # calculate averages for AUCs and pAUCs
            averaged_performance = np.mean(np.array(performance, dtype=float), axis=0)
            mean_auc, mean_p_auc = averaged_performance[0], averaged_performance[1]
            self.logger.info(f'{machine_type}\t\tAUC: {mean_auc * 100:.3f}\tpAUC: {mean_p_auc * 100:.3f}')

            sum_auc += mean_auc
            sum_pauc += mean_p_auc
            num += 1
        avg_auc, avg_pauc = sum_auc / num, sum_pauc / num
        # print(f'Total average:\t\tAUC: {avg_auc * 100:.3f}\tpAUC: {avg_pauc * 100:.3f}')
        self.logger.info(f'Total average:\t\tAUC: {avg_auc * 100:.3f}\tpAUC: {avg_pauc * 100:.3f}')
        return avg_auc, avg_pauc


    def valid_mcmc(self, save=False, eval_iteration=3):

        sum_auc, sum_pauc, num = 0, 0, 0
        result_dir = os.path.join(self.args.result_dir, self.args.version)

        
        
        net = self.net.module if self.args.dp else self.net
        print('\n' + '=' * 20)

        for index, (target_dir, train_dir) in enumerate(
                zip(sorted(self.args.valid_dirs), sorted(self.args.train_dirs))):
            machine_type = target_dir.split('/')[-2]
            # result csv

            performance = []
            # get machine list
            machine_id_list = utils.get_machine_id_list(target_dir)
            for id_str in machine_id_list:
                test_files, y_true = utils.create_test_file_list(target_dir, id_str, dir_name='test')
                y_pred = []
                valid_dataset = ASDDataset(self.args, test_files, load_in_memory=False)
                valid_dataloader = DataLoader(valid_dataset, batch_size=self.args.batch_size,
                                               shuffle=False, num_workers=self.args.num_workers)

                for i,(x_wav, x_mel, label) in enumerate(valid_dataloader):
                    x_wav, x_mel = x_wav.float().to(self.args.device), x_mel.float().to(self.args.device)
                    x_mel = x_mel.unsqueeze(1)
                    x_wav = x_wav.unsqueeze(1)

                    label = label.long().to(self.args.device)
                    
                    with torch.no_grad():
                        anomaly_score = 0
                        for i in range(eval_iteration):
                            svdd_anomaly_score = net.compute_anomaly_score(x_mel, x_wav, label)
                            anomaly_score += svdd_anomaly_score
                        anomaly_score = anomaly_score/eval_iteration
                    anomaly_score = anomaly_score.squeeze().cpu().numpy()
                    

                    y_pred.extend(anomaly_score)

                # compute auc and pAuc
                max_fpr = 0.1
                auc = sklearn.metrics.roc_auc_score(y_true, y_pred)
                p_auc = sklearn.metrics.roc_auc_score(y_true, y_pred, max_fpr=max_fpr)
                performance.append([auc, p_auc])

            # calculate averages for AUCs and pAUCs
            averaged_performance = np.mean(np.array(performance, dtype=float), axis=0)
            mean_auc, mean_p_auc = averaged_performance[0], averaged_performance[1]
            print(f'{machine_type}\t\tAUC: {mean_auc * 100:.3f}\tpAUC: {mean_p_auc * 100:.3f}')

            sum_auc += mean_auc
            sum_pauc += mean_p_auc
            num += 1
        avg_auc, avg_pauc = sum_auc / num, sum_pauc / num
        # print(f'Total average:\t\tAUC: {avg_auc * 100:.3f}\tpAUC: {avg_pauc * 100:.3f}')
        self.logger.info(f'Total average:\t\tAUC: {avg_auc * 100:.3f}\tpAUC: {avg_pauc * 100:.3f}')
        return avg_auc, avg_pauc

        
    


