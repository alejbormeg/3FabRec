from constants import TRAIN, VAL
import time
import datetime
import os
import pandas as pd
import numpy as np
from csl_common.utils import nn as nn
import torch
import torch.utils.data as td
import torch.nn.modules.distance
import torch.optim as optim
import torch.nn.functional as F
import config as cfg
import csl_common.utils.ds_utils as ds_utils
from datasets import wflw, w300, aflw, forense_am
from csl_common.utils import log
from csl_common.utils.nn import to_numpy, Batch
from train_aae_unsupervised import AAETraining
from landmarks import lmutils, lmvis, fabrec
import landmarks.lmconfig as lmcfg
import aae_training
import matplotlib.pyplot as plt

class AAELandmarkTraining(AAETraining):

    def __init__(self, datasets, args, session_name='debug', **kwargs):
        args.reset = False  # just to make sure we don't reset the discriminator by accident
        try:
            ds = datasets[TRAIN]
        except KeyError:
            ds = datasets[VAL]
        self.num_landmarks = ds.NUM_LANDMARKS
        self.all_landmarks = ds.ALL_LANDMARKS
        self.landmarks_no_outline = ds.LANDMARKS_NO_OUTLINE
        self.landmarks_only_outline = ds.LANDMARKS_ONLY_OUTLINE
        self.first_time = True
        super().__init__(datasets, args, session_name, macro_batch_size=0, **kwargs)

        self.optimizer_lm_head = optim.Adam(self.saae.LMH.parameters(), lr=args.lr_heatmaps, betas=(0.9, 0.999))
        self.optimizer_E = optim.Adam(self.saae.Q.parameters(), lr=0.00002, betas=(0.9, 0.999))
        # self.optimizer_G = optim.Adam(self.saae.P.parameters(), lr=0.00002, betas=(0.9, 0.999)) #No se define para G por el FineTuning
        self.output_stats = pd.DataFrame(
            columns=['Época', 'Segundos por época', 'Error de reconstrucción', 'Distancia L2 entren Heathmaps'])

        self.eval_stats_image = pd.DataFrame(
            columns=['ID imagen', 'Media NME', 'Error de Reconstrucción']
        )
        self.eval_stats_landmark = pd.DataFrame(
            columns=['Landmark', 'Media NME por landmark']
        )
        self.reconstruction_errors_val=[]
        self.images_nmes_train=[]
        self.images_nmes_eval=[]
        self.train_nmes=[]
        self.eval_nmes=[]
        self.num_epochs=0
        self.total_epochs=0;

    def _get_network(self, pretrained):
        return fabrec.Fabrec(self.num_landmarks, input_size=self.args.input_size, z_dim=self.args.embedding_dims)

    def create_batches_for_true_and_predicted_landmarks(self, batch, lm_preds_max, data):
        lm_predict = []
        true_lm = []
        batch_true = []
        batch_pred = []
        # Recorremos cada imagen
        cont_imagenes = 0
        for v in batch.landmarks:
            cont_landmarks = 0
            # recorremos mapas de calor
            for i in v:
                if data['mask'][cont_imagenes][cont_landmarks] == 1:
                    lm_predict.append(lm_preds_max[cont_imagenes][cont_landmarks])
                    true_lm.append(i.cpu().numpy())
                else:
                    true_lm.append([0, 0])
                    lm_predict.append([0, 0])

                cont_landmarks += 1
            cont_imagenes += 1
            batch_true.append(true_lm)
            batch_pred.append(lm_predict)
            true_lm = []
            lm_predict = []

        batch_pred = torch.Tensor(batch_pred)
        batch_true = torch.Tensor(batch_true)
        return batch_pred, batch_true

    @staticmethod
    def print_eval_metrics(nmes, show=False):
        def ced_curve(_nmes):
            y = []
            x = np.linspace(0, 10, 50)
            for th in x:
                recall = 1.0 - lmutils.calc_landmark_failure_rate(_nmes, th)
                recall *= 1 / len(x)
                y.append(recall)
            return x, y

        def auc(recalls):
            return np.sum(recalls)

        # for err_scale in np.linspace(0.1, 1, 10):
        for err_scale in [1.0]:
            # print('\nerr_scale', err_scale)
            # print(np.clip(lm_errs_max_all, a_min=0, a_max=10).mean())

            fr = lmutils.calc_landmark_failure_rate(nmes * err_scale)
            X, Y = ced_curve(nmes)

            log.info('NME:   {:>6.3f}'.format(nmes.mean() * err_scale))
            log.info('FR@10: {:>6.3f} ({})'.format(fr * 100, np.sum(nmes.mean(axis=1) > 10)))
            log.info('AUC:   {:>6.4f}'.format(auc(Y)))
            # log.info('NME:   {nme:>6.3f}, FR@10: {fr:>6.3f} ({fc}), AUC:   {auc:>6.4f}'.format(
            #     nme=nmes.mean()*err_scale,
            #     fr=fr*100,
            #     fc=np.sum(nmes.mean(axis=1) > 10),
            #     auc=auc(Y)))

            if show:
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(1, 2)
                axes[0].plot(X, Y)
                axes[1].hist(nmes.mean(axis=1), bins=20)
                plt.show()

    def _print_iter_stats(self, stats):
        means = pd.DataFrame(stats).mean().to_dict()
        current = stats[-1]
        nmes = current.get('nmes', np.zeros(0))

        str_stats = ['[{ep}][{i}/{iters_per_epoch}] '
                     'l_rec={avg_loss_recon:.3f} '
                     'ssim={avg_ssim:.3f} '
                     # 'ssim_torch={avg_ssim_torch:.3f} '
                     # 'z_mu={avg_z_recon_mean: .3f} '
                     'l_lms={avg_loss_lms:.4f} '
                     'err_lms={avg_err_lms:.2f}/{avg_err_lms_outline:.2f}/{avg_err_lms_all:.2f} '
                     '{t_data:.2f}/{t_proc:.2f}/{t:.2f}s ({total_iter:06d} {total_time})'][0]

        log.info(str_stats.format(
            ep=current['epoch'] + 1, i=current['iter'] + 1, iters_per_epoch=self.iters_per_epoch,
            avg_loss=means.get('loss', -1),
            avg_loss_recon=means.get('loss_recon', -1),
            avg_ssim=1.0 - means.get('ssim', -1),
            avg_ssim_torch=means.get('ssim_torch', -1),
            avg_loss_activations=means.get('loss_activations', -1),
            avg_loss_lms=means.get('loss_lms', -1),
            avg_z_l1=means.get('z_l1', -1),
            avg_z_recon_mean=means.get('z_recon_mean', -1),
            t=means['iter_time'],
            t_data=means['time_dataloading'],
            t_proc=means['time_processing'],
            avg_err_lms=nmes[:, self.landmarks_no_outline].mean(),
            avg_err_lms_outline=nmes[:, self.landmarks_only_outline].mean(),
            avg_err_lms_all=nmes[:, self.all_landmarks].mean(),
            total_iter=self.total_iter + 1, total_time=str(datetime.timedelta(seconds=self._training_time()))
        ))

    def create_learning_curves(self,partition=''):
        epochs=range(1,int(self.total_epochs)+1)

        plt.plot(epochs,self.train_nmes,color='orange')
        plt.plot(epochs,self.eval_nmes,color='blue')
        plt.legend(['Curva de etrenamiento','Curva de validación'])
        plt.title("Curvas de aprendizaje "+partition)
        plt.xlabel('Épocas')
        plt.ylabel('Media NME')
        plt.savefig("./data/Outputs/curvas_aprendizaje.png")

    def compute_mean(selfself, v,partition=''):
        i = 0
        sum = 0
        for elem in v:
            if elem != 0.0:
                i += 1
                sum += elem
        return round(sum / i,3)

    def fill_eval_stats(self, nmes):
        i = 0
        for r in nmes:
            data_image = {
                'ID imagen': i,
                'Media NME': self.compute_mean(r),
                'Error de Reconstrucción': round(float(self.reconstruction_errors_val[i]),3)
            }
            i += 1
            self.eval_stats_image = self.eval_stats_image.append(data_image, ignore_index=True)

        nmes_transpose = list(map(list, zip(*nmes)))
        i=0
        for r in nmes_transpose:
            data_landmark={
                'Landmark': lmutils.landmarks[i],
                'Media NME por landmark': self.compute_mean(r)
            }
            i+=1
            self.eval_stats_landmark = self.eval_stats_landmark.append(data_landmark, ignore_index=True)

    def _print_epoch_summary(self, epoch_stats, epoch_starttime, eval=False):
        means = pd.DataFrame(epoch_stats).mean().to_dict()
        """
        try:
            nmes = np.concatenate([s['nmes'] for s in self.epoch_stats if 'nmes' in s])
        except KeyError:
            nmes = np.zeros((1,100))
        """

        if eval:
            nmes = np.concatenate([s['nmes'] for s in self.epoch_stats if 'nmes' in s])
            self.fill_eval_stats(nmes)
        else:
            nmes = np.zeros((1, 100))

        duration = int(time.time() - epoch_starttime)
        log.info("{}".format('-' * 100))
        str_stats = ['           '
                     'l_rec={avg_loss_recon:.3f} '
                     'ssim={avg_ssim:.3f} '
                     # 'ssim_torch={avg_ssim_torch:.3f} '
                     # 'z_mu={avg_z_recon_mean:.3f} '
                     'l_lms={avg_loss_lms:.4f} '
                     'err_lms={avg_err_lms:.2f}/{avg_err_lms_outline:.2f}/{avg_err_lms_all:.2f} '
                     '\tT: {time_epoch}'][0]
        data = {
            'Epochs': means.get('epoch'),
            'Seconds per epoch': duration,
            'Reconstruction Error': round(means.get('loss_recon'),3),
            'Normalized Mean Error for landmarks': round(means.get('loss_lms'),3)
        }
        self.output_stats = self.output_stats.append(data, ignore_index=True)
        log.info(str_stats.format(
            iters_per_epoch=self.iters_per_epoch,
            avg_loss=means.get('loss', -1),
            avg_loss_recon=means.get('loss_recon', -1),
            avg_ssim=1.0 - means.get('ssim', -1),
            avg_ssim_torch=means.get('ssim_torch', -1),
            avg_loss_lms=means.get('loss_lms', -1),
            avg_loss_lms_cnn=means.get('loss_lms_cnn', -1),
            avg_err_lms=nmes[:, self.landmarks_no_outline].mean(),
            avg_err_lms_outline=nmes[:, self.landmarks_only_outline].mean(),
            avg_err_lms_all=nmes[:, self.all_landmarks].mean(),
            avg_z_recon_mean=means.get('z_recon_mean', -1),
            t=means['iter_time'],
            t_data=means['time_dataloading'],
            t_proc=means['time_processing'],
            total_iter=self.total_iter + 1, total_time=str(datetime.timedelta(seconds=self._training_time())),
            time_epoch=str(datetime.timedelta(seconds=duration))))
        try:
            recon_errors = np.concatenate([stats['l1_recon_errors'] for stats in self.epoch_stats])
            rmse = np.sqrt(np.mean(recon_errors ** 2))
            log.info("RMSE: {} ".format(rmse))
        except KeyError:
            # print("no l1_recon_error")
            pass

        if self.args.eval and nmes is not None:
            # benchmark_mode = hasattr(self.args, 'benchmark')
            # self.print_eval_metrics(nmes, show=benchmark_mode)
            self.print_eval_metrics(nmes, show=True)

    def eval_epoch(self, filename="",num_epochs=0):
        log.info("")
        log.info("Evaluating '{}'...".format(self.session_name))
        # log.info("")

        epoch_starttime = time.time()
        self.epoch_stats = []
        self.saae.eval()

        self._run_epoch(self.datasets[VAL], eval=True, filename=filename)
        # print average loss and accuracy over epoch
        self._print_epoch_summary(self.epoch_stats, epoch_starttime, eval=True)

        return self.epoch_stats

    def train(self, num_epochs=None, partition=''):

        log.info("")
        log.info("Starting training session '{}'...".format(self.session_name))
        # log.info("")
        self.num_epochs=num_epochs
        while num_epochs is None or self.epoch < num_epochs:
            log.info('')
            log.info('Epoch {}/{}'.format(self.epoch + 1, num_epochs))
            log.info('=' * 10)

            self.epoch_stats = []
            epoch_starttime = time.time()

            self._run_epoch(self.datasets[TRAIN])
            #TODO Revisar si esto falla
            self.eval_epoch(filename=partition + 'eval')
            self.snapshot_interval = args.save_freq
            # save model every few epochs
            if (self.epoch + 1) % self.snapshot_interval == 0:
                log.info("*** saving snapshot *** ")
                self._save_snapshot(is_best=False, partition=partition)

            # print average loss and accuracy over epoch
            self._print_epoch_summary(self.epoch_stats, epoch_starttime)

            """
            if self._is_eval_epoch():
                self.eval_epoch()
            """
            #TODO Esto comentado era como estaba antes el framework
            '''
            if self.epoch + 1 == num_epochs:
                self.eval_epoch(filename=partition + 'eval')
            '''

            self.epoch += 1
            self.total_epochs+=1
        # Save output stats
        self.output_stats.to_csv("./data/Outputs/" + partition + ".csv")
        self.eval_stats_image.to_csv("./data/Outputs/" + partition+"_eval_images" + ".csv")
        self.eval_stats_landmark.to_csv("./data/Outputs/" + partition+"_eval_landmark" + ".csv")

        time_elapsed = time.time() - self.time_start_training

        #TODO Revisar, calculamos curvas de aprendizaje
        self.create_learning_curves(partition)
        log.info('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    def _run_epoch(self, dataset, eval=False, filename=""):
        batchsize = self.args.batchsize_eval if eval else self.args.batchsize
        self.iters_per_epoch = int(len(dataset) / batchsize)
        self.iter_starttime = time.time()
        self.iter_in_epoch = 0

        dataloader = td.DataLoader(dataset, batch_size=batchsize, shuffle=False,
                                   num_workers=self.workers, drop_last=False)
        for data in dataloader:
            self.total_iter += 1
            self.iter_in_epoch += 1
            self._run_batch(data, eval=eval, filename=filename + str(self.iter_in_epoch))
            self.saae.total_iter = self.total_iter

        #TODO Revisar, calculamos los nmes para curvas de aprendizaje
        if eval:
            sum=0
            for elem in self.images_nmes_eval:
                sum+=elem
            self.eval_nmes.append(sum/len(self.images_nmes_eval))
            self.images_nmes_eval=[]
        else:
            sum=0
            for elem in self.images_nmes_train:
                sum+=elem
            self.train_nmes.append(sum/len(self.images_nmes_train))
            self.images_nmes_train=[]




    def _run_batch(self, data, eval=False, ds=None, filename=""):
        time_dataloading = time.time() - self.iter_starttime
        time_proc_start = time.time()
        iter_stats = {'time_dataloading': time_dataloading}
        batch = Batch(data, eval=eval)
        # Gradientes a 0
        self.saae.zero_grad()
        self.saae.eval()  # Pone la red en Modo evaluación

        input_images = batch.target_images if batch.target_images is not None else batch.images  # carga las imágenes

        with torch.set_grad_enabled(self.args.train_encoder):  # Si estamos entrenando el encoder calcula gradientes
            z_sample = self.saae.Q(input_images)  # Calcula el embedding

        iter_stats.update({'z_recon_mean': z_sample.mean().item()})

        #######################
        # Reconstruction phase
        #######################
        with torch.set_grad_enabled(self.args.train_encoder and not eval):
            X_recon = self.saae.P(z_sample)

        # calculate reconstruction error for debugging and reporting
        with torch.no_grad():
            iter_stats['loss_recon'] = aae_training.loss_recon(batch.images, X_recon)

        #######################
        # Landmark predictions
        #######################
        train_lmhead = not eval  # Entrenamos cuando no estamos en modo eval
        lm_preds_max = None
        with torch.set_grad_enabled(train_lmhead):
            self.saae.LMH.train(train_lmhead)  # Si train_lmhead=True ponemos modo entrenamiento
            X_lm_hm = self.saae.LMH(self.saae.P)  # Obtiene los mapas de calor

            if batch.lm_heatmaps is not None:
                lm_hm_predict = []
                true_lm_hm = []
                # Recorremos cada imagen
                cont_imagenes = 0
                for v in batch.lm_heatmaps:
                    cont_mapas_calor = 0
                    # recorremos mapas de calor
                    for i in v:
                        if data['mask'][cont_imagenes][cont_mapas_calor] == 1:
                            lm_hm_predict.append(X_lm_hm[cont_imagenes][cont_mapas_calor].cpu())
                            true_lm_hm.append(i.cpu().numpy())
                        cont_mapas_calor += 1
                    cont_imagenes += 1

                lm_hm_predict = torch.stack(lm_hm_predict)
                true_lm_hm = np.array(true_lm_hm)
                true_lm_hm = torch.from_numpy(true_lm_hm)
                loss_lms = F.mse_loss(true_lm_hm,
                                      lm_hm_predict) * 100 * 3  # Calcula la distancia L2 entre los mapas de calor
                iter_stats.update({'loss_lms': loss_lms.item()})

            #TODO revisar esta sección, pues cambia mucho cómo funciona el framework
            X_lm_hm = lmutils.smooth_heatmaps(X_lm_hm)
            lm_preds_max = self.saae.heatmaps_to_landmarks(X_lm_hm)
            batch_pred, batch_true = self.create_batches_for_true_and_predicted_landmarks(batch, lm_preds_max, data)
            nmes = lmutils.calc_landmark_nme(batch_pred, batch_true, ocular_norm=self.args.ocular_norm,
                                             image_size=self.args.input_size)

            iter_stats.update({'nmes': nmes})

            if not eval:
                for v in nmes:
                    self.images_nmes_train.append(self.compute_mean(v))
            else:
                for v in nmes:
                    self.images_nmes_eval.append(self.compute_mean(v))

            '''
            if eval or self._is_printout_iter(eval):
                X_lm_hm = lmutils.smooth_heatmaps(X_lm_hm)
                lm_preds_max = self.saae.heatmaps_to_landmarks(X_lm_hm)

            if eval or self._is_printout_iter(eval):
                batch_pred, batch_true = self.create_batches_for_true_and_predicted_landmarks(batch, lm_preds_max, data)
                nmes = lmutils.calc_landmark_nme(batch_pred, batch_true, ocular_norm=self.args.ocular_norm,
                                                 image_size=self.args.input_size)
                iter_stats.update({'nmes': nmes})
            '''

        if train_lmhead:
            # if self.args.train_encoder:
            #     loss_lms = loss_lms * 80.0
            loss_lms.backward()  # Calcula gradientes
            self.optimizer_lm_head.step()  # Hace Backprop por la red
            if self.args.train_encoder:
                self.optimizer_E.step()  # para el fine tuning
                # self.optimizer_G.step()

        # statistics
        iter_stats.update({'epoch': self.epoch, 'timestamp': time.time(),
                           'iter_time': time.time() - self.iter_starttime,
                           'time_processing': time.time() - time_proc_start,
                           'iter': self.iter_in_epoch, 'total_iter': self.total_iter, 'batch_size': len(batch)})
        self.iter_starttime = time.time()
        self.epoch_stats.append(iter_stats)

        # print stats every N mini-batches
        if self._is_printout_iter(eval):
            images=nn.atleast4d(batch.images)
            reconstruction=255.0 * torch.abs(images - X_recon).reshape(len(images), -1).mean(dim=1)
            for i in reconstruction:
                self.reconstruction_errors_val.append(i.cpu().numpy())
            self._print_iter_stats(self.epoch_stats[-self._print_interval(eval):])
            if self.epoch +1 == self.num_epochs:
                lmvis.visualize_batch(batch.images, batch_true, X_recon, X_lm_hm, batch_pred,
                                      lm_heatmaps=batch.lm_heatmaps,
                                      target_images=batch.target_images,
                                      ds=ds,
                                      ocular_norm=self.args.ocular_norm,
                                      clean=False,
                                      overlay_heatmaps_input=False,
                                      overlay_heatmaps_recon=False,
                                      landmarks_only_outline=self.landmarks_only_outline,
                                      landmarks_no_outline=self.landmarks_no_outline,
                                      f=1.0,
                                      wait=self.wait,
                                      draw_gt_offsets=False,
                                      filename=filename)

def basemodel():
    from csl_common.utils.common import init_random

    if args.seed is not None:
        init_random(args.seed)
    # log.info(json.dumps(vars(args), indent=4))
    datasets = {}
    # Cross validation 5-fold
    for i in range(5):
        for phase, dsnames, num_samples in zip((TRAIN, VAL),
                                               (args.dataset_train, args.dataset_val),
                                               (args.train_count, args.val_count)):
            train = phase == TRAIN
            name = dsnames[0]
            # transform = ds_utils.build_transform(deterministic=not train, daug=6)
            transform = None
            root, cache_root = cfg.get_dataset_paths(name)
            dataset_cls = cfg.get_dataset_class(name)
            if phase == TRAIN:
                datasets[phase] = dataset_cls(root=root,
                                              cache_root=cache_root,
                                              train=train,
                                              max_samples=num_samples,
                                              use_cache=args.use_cache,
                                              start=args.st,
                                              test_split=None,
                                              align_face_orientation=args.align,
                                              crop_source=args.crop_source,
                                              return_landmark_heatmaps=lmcfg.PREDICT_HEATMAP,
                                              with_occlusions=args.occ and train,
                                              landmark_sigma=args.sigma,
                                              transform=transform,
                                              image_size=args.input_size,
                                              cross_val_split=i + 1,
                                              )
            else:
                datasets[phase] = dataset_cls(root=root,
                                              cache_root=cache_root,
                                              train=train,
                                              max_samples=num_samples,
                                              use_cache=args.use_cache,
                                              start=args.st,
                                              test_split='test',
                                              align_face_orientation=args.align,
                                              crop_source=args.crop_source,
                                              return_landmark_heatmaps=lmcfg.PREDICT_HEATMAP,
                                              with_occlusions=args.occ and train,
                                              landmark_sigma=args.sigma,
                                              transform=transform,
                                              image_size=args.input_size,
                                              cross_val_split=i + 1,
                                              )

        fntr = AAELandmarkTraining(datasets, args, session_name=args.sessionname, snapshot_interval=args.save_freq,
                                   workers=args.workers, wait=args.wait)

        torch.backends.cudnn.benchmark = True
        if args.eval:
            fntr.eval_epoch()
        else:
            fntr.train(num_epochs=args.epochs, partition='/partition_' + str(i + 1))


def bm_daug():
    from csl_common.utils.common import init_random

    if args.seed is not None:
        init_random(args.seed)

    args.sessionname = "DataAugmentation"
    resume = args.resume
    # log.info(json.dumps(vars(args), indent=4))
    datasets = {}
    # Cross validation 5-fold
    for i in range(5):
        for phase, dsnames, num_samples in zip((TRAIN, VAL),
                                               (args.dataset_train, args.dataset_val),
                                               (args.train_count, args.val_count)):
            train = phase == TRAIN
            name = dsnames[0]
            # transform = ds_utils.build_transform(deterministic=not train, daug=6)
            transform = None
            root, cache_root = cfg.get_dataset_paths(name)
            dataset_cls = cfg.get_dataset_class(name)
            args.resume = resume
            if phase == TRAIN:
                datasets[phase] = dataset_cls(root=root,
                                              cache_root=cache_root,
                                              train=train,
                                              max_samples=num_samples,
                                              use_cache=args.use_cache,
                                              start=args.st,
                                              test_split=None,
                                              align_face_orientation=args.align,
                                              crop_source=args.crop_source,
                                              return_landmark_heatmaps=lmcfg.PREDICT_HEATMAP,
                                              with_occlusions=args.occ and train,
                                              landmark_sigma=args.sigma,
                                              transform=transform,
                                              image_size=args.input_size,
                                              cross_val_split=i + 1,
                                              )
            else:
                datasets[phase] = dataset_cls(root=root,
                                              cache_root=cache_root,
                                              train=train,
                                              max_samples=num_samples,
                                              use_cache=args.use_cache,
                                              start=args.st,
                                              test_split='test',
                                              align_face_orientation=args.align,
                                              crop_source=args.crop_source,
                                              return_landmark_heatmaps=lmcfg.PREDICT_HEATMAP,
                                              with_occlusions=args.occ and train,
                                              landmark_sigma=args.sigma,
                                              transform=transform,
                                              image_size=args.input_size,
                                              cross_val_split=i + 1,
                                              )

        fntr = AAELandmarkTraining(datasets, args, session_name=args.sessionname, snapshot_interval=args.save_freq,
                                   workers=args.workers, wait=args.wait)

        torch.backends.cudnn.benchmark = True
        if args.eval:
            fntr.eval_epoch()
        else:
            fntr.train(num_epochs=args.epochs, partition='/partition_' + str(i + 1))
            transform = ds_utils.build_transform(deterministic=not train, daug=6)
            datasets[TRAIN] = dataset_cls(root=root,
                                          cache_root=cache_root,
                                          train=train,
                                          max_samples=num_samples,
                                          use_cache=args.use_cache,
                                          start=args.st,
                                          test_split='test',
                                          align_face_orientation=args.align,
                                          crop_source=args.crop_source,
                                          return_landmark_heatmaps=lmcfg.PREDICT_HEATMAP,
                                          with_occlusions=args.occ and train,
                                          landmark_sigma=args.sigma,
                                          transform=transform,
                                          image_size=args.input_size,
                                          cross_val_split=i + 1)

            args.resume = "DataAugmentation/partition_" + str(i + 1) + "/00" + str(args.epochs)
            daug_1 = AAELandmarkTraining(datasets, args, session_name="Daug_1", snapshot_interval=args.save_freq,
                                         workers=args.workers, wait=args.wait)
            daug_1.train(num_epochs=args.epochs + 20, partition='/partition_' + str(i + 1) + 'daug_1')

            transform = ds_utils.build_transform(deterministic=not train, daug=8)
            datasets[TRAIN] = dataset_cls(root=root,
                                          cache_root=cache_root,
                                          train=train,
                                          max_samples=num_samples,
                                          use_cache=args.use_cache,
                                          start=args.st,
                                          test_split='test',
                                          align_face_orientation=args.align,
                                          crop_source=args.crop_source,
                                          return_landmark_heatmaps=lmcfg.PREDICT_HEATMAP,
                                          with_occlusions=args.occ and train,
                                          landmark_sigma=args.sigma,
                                          transform=transform,
                                          image_size=args.input_size,
                                          cross_val_split=i + 1)

            args.resume = "Daug_1/partition_" + str(i + 1) + "daug_1" + "/00" + str(args.epochs + 20)
            daug_2 = AAELandmarkTraining(datasets, args, session_name="Daug_2",
                                         snapshot_interval=args.save_freq,
                                         workers=args.workers, wait=args.wait)
            daug_2.train(num_epochs=args.epochs + 40, partition='/partition_' + str(i + 1) + 'daug_2')


def fine_tuning():
    print("Hola")


def run():
    basemodel()
    # bm_daug()
    fine_tuning()


if __name__ == '__main__':

    import sys
    import configargparse

    np.set_printoptions(linewidth=np.inf)

    # Disable traceback on Ctrl+c
    import signal

    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    parser = configargparse.ArgParser()
    defaults = {
        'batchsize': 30,
        'train_encoder': False,
        'train_decoder': False
    }
    aae_training.add_arguments(parser, defaults)

    # Dataset
    parser.add_argument('--dataset', default=['w300'], type=str, help='dataset for training and testing',
                        choices=['w300', 'aflw', 'wflw', 'forense_am', 'forense_am_test'], nargs='+')
    parser.add_argument('--test-split', default='full', type=str, help='test set split for 300W/AFLW/WFLW',
                        choices=['challenging', 'common', '300w', 'full', 'frontal'] + wflw.SUBSETS)

    # Landmarks
    parser.add_argument('--lr-heatmaps', default=0.001, type=float, help='learning rate for landmark heatmap outputs')
    parser.add_argument('--sigma', default=7, type=float, help='size of landmarks in heatmap')
    parser.add_argument('-n', '--ocular-norm', default=lmcfg.LANDMARK_OCULAR_NORM, type=str,
                        help='how to normalize landmark errors', choices=['pupil', 'outer', 'none'])

    args = parser.parse_args()

    args.dataset_train = args.dataset
    args.dataset_val = args.dataset

    if args.sessionname is None:
        if args.resume:
            modelname = os.path.split(args.resume)[0]
            args.sessionname = modelname
        else:
            args.sessionname = 'debug'

    run()
