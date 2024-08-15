import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk

import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import cv2
from PIL import Image


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        # output_model = [model_output == i for i in range(1, 4)]
        # class_rv = np.array(output_model[0].cpu()).astype(np.uint8)
        # class_myo = np.array(output_model[1].cpu()).astype(np.uint8)
        # class_lv = np.array(output_model[2].cpu()).astype(np.uint8)
        # img = torch.from_numpy(class_rv).permute(1, 2, 0)
        # img = img.detach().numpy()
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        # img = img.astype(np.float32)
        # plt.imshow(img)
        # plt.show()
        return (model_output[self.category, :, :] * self.mask).sum()

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0

def mask_classes(mask, num_classes = 3):
    _mask = [mask == i for i in range(1, num_classes + 1)]
    # _mask = torch.tensor([item.cpu().detach().numpy() for item in _mask]).cuda()
    # _mask = _mask.cpu().numpy().astype(np.uint8)
    class_rv = np.array(_mask[0].cpu()).astype(np.uint8)
    class_myo = np.array(_mask[1].cpu()).astype(np.uint8)
    class_lv = np.array(_mask[2].cpu()).astype(np.uint8)
    # class_xx = np.array(_mask[3].cpu()).astype(np.uint8)
    # img = torch.from_numpy(class_lv.squeeze(0)).permute(1, 2, 0)
    # img = img.detach().numpy()
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # img = img.astype(np.float32)
    # plt.imshow(img)
    # plt.show()
    return class_rv, class_myo, class_lv

def reshape_transform(tensor, height=224, width=224):
    '''
    不同参数的Swin网络的height和width是不同的，具体需要查看所对应的配置文件yaml
    height = width = config.DATA.IMG_SIZE / config.MODEL.NUM_HEADS[-1]
    比如该例子中IMG_SIZE: 224  NUM_HEADS: [4, 8, 16, 32]
    height = width = 224 / 32 = 7
    '''
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    metric_all_slice = []       # gao
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            # ind = 100
            slice = image[ind, :, :]
            slice_label = label[ind, :, :]      # gao
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()

            # CAM gao 1
            #
            # target_layers = [net.swin_unet.up.norm]
            # cam = GradCAM(model=net, target_layers=target_layers,
            #               use_cuda=True, reshape_transform=reshape_transform)
            # # CAM gao 1
            # # slice_label = label[ind, :, :]            # label slice
            # # x, y = slice_label.shape[0], slice_label.shape[1]
            # # if x != patch_size[0] or y != patch_size[1]:
            # #     slice_label = zoom(slice_label, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            # # label_three = torch.from_numpy(slice_label).unsqueeze(0).unsqueeze(0).float().cuda()
            # # rv, myo, lv = mask_classes(label_three)
            # # cam_label = [rv.squeeze(0), myo.squeeze(0), lv.squeeze(0)]
            #
            # # CAM gao
            # input_tensor = torch.from_numpy(slice).unsqueeze(2).float()
            # input_tensor = input_tensor.numpy()
            # # input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_GRAY2RGB)
            # input_tensor = np.concatenate((input_tensor, input_tensor, input_tensor), axis=-1)
            # input_tensor = input_tensor * 255
            # input_tensor = cv2.normalize(input_tensor, None, 0, 255, cv2.NORM_MINMAX)
            # # input_tensor = np.array(input_tensor, dtype='uint8')
            # input_tensor = input_tensor.astype(np.uint8) / 255
            #
            # mask_test = net(input)
            # normalized_masks = torch.nn.functional.softmax(mask_test, dim=1)  #.cpu
            # # normalized_masks = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            # # sem_classes = [
            # #     '__background__', 'rv', 'myo', 'lv'
            # # ]
            # sem_classes = [
            #     '__background__', 'Aorta', 'Gallbladder', 'Kidney(L)', 'Kidney(R)', 'Liver', 'Pancreas', 'Spleen',
            #     'Stomach'
            # ]
            # sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}
            #
            # car_category = sem_class_to_idx["Aorta"]
            # car_mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
            # car_mask_uint8 = 255 * np.uint8(car_mask == car_category)
            # car_mask_float = np.float32(car_mask == car_category)
            #
            # # both_images = np.hstack((input_tensor, np.repeat(car_mask_uint8[:, :, None], 3, axis=-1)))
            # # Image.fromarray(both_images)
            # # plt.imshow(both_images)
            # # plt.show()
            #
            # # down = nn.Linear(1, 3)
            # # image2 = input_tensor.squeeze(0).cpu().permute(1, 2, 0)         # GRAY2RGB
            # # image2 = image2.detach().numpy()
            # # image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2RGB)
            # # image2 = image2.astype(np.float32) / 255
            # # plt.imshow(input_tensor)
            # # plt.show()
            #
            # # cam.batch_size = 1
            #
            # # class_map = {0: 'Background', 1: 'RV', 2: 'Myo', 3: 'LV'}
            # class_map = {0: 'Background', 1: 'Aorta', 2: 'Gallbladder', 3: 'Kidney(L)', 4: 'Kidney(R)', 5: 'Liver', 6: 'Pancreas',
            #              7: 'Spleen', 8: 'Stomach'}
            # save_path = "D:\Project_fan\Swin-Unet-Stabl_BGM\plot/Synapse"
            # for i in range(9):
            #     class_id = i
            #     class_name = class_map[class_id]
            #     # [SemanticSegmentationTarget(class_id, mask=cam_label[class_id - 1])]
            #     targets = [SemanticSegmentationTarget(i, car_mask_float)]
            #
            #     grayscale_cam = cam(input_tensor=input,
            #                         targets=targets,
            #                         # targets=targets,
            #                         eigen_smooth=False,
            #                         aug_smooth=False)
            #
            #     # Here grayscale_cam has only one image in the batch
            #     grayscale_cam = grayscale_cam[0, :]
            #
            #     cam_image = show_cam_on_image(input_tensor, grayscale_cam, use_rgb=True)
            #     plt.imshow(cam_image)
            #     # plt.title(class_name)
            #     plt.axis('off')  # 去坐标轴
            #     plt.xticks([])  # 去 x 轴刻度
            #     plt.yticks([])  # 去 y 轴刻度
            #     plt.savefig(save_path + "/bgm+ds_slice_100_{}".format(i), bbox_inches='tight', pad_inches=0)
            #     plt.show()
            # CAM gao

            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
                # p = []
                # for i in range(1, classes):
                #     p.append(calculate_metric_percase(pred == i, slice_label == i))
                # performance = np.mean(p, axis=0)[0]
                # mean_hd95 = np.mean(p, axis=0)[1]
                # metric_all_slice.append([performance, mean_hd95])      # gao
                # print(metric_all_slice)
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    # p = []
    # p.append(calculate_metric_percase(prediction == 6, label == 6))
    # performance = np.mean(p, axis=0)[0]
    # mean_hd95 = np.mean(p, axis=0)[1]
    # metric_all_slice.append([performance, mean_hd95])
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    ## Synapse for AMOS
    # metric_list.append(calculate_metric_percase(prediction == 1, label == 8))
    # metric_list.append(calculate_metric_percase(prediction == 2, label == 4))
    # metric_list.append(calculate_metric_percase(prediction == 3, label == 3))
    # metric_list.append(calculate_metric_percase(prediction == 4, label == 2))
    # metric_list.append(calculate_metric_percase(prediction == 5, label == 6))
    # metric_list.append(calculate_metric_percase(prediction == 6, label == 10))
    # metric_list.append(calculate_metric_percase(prediction == 7, label == 1))
    # metric_list.append(calculate_metric_percase(prediction == 8, label == 7))
    ##
    # metric_list.append(calculate_metric_percase(prediction == 8, label == 1))
    # metric_list.append(calculate_metric_percase(prediction == 4, label == 2))
    # metric_list.append(calculate_metric_percase(prediction == 3, label == 3))
    # metric_list.append(calculate_metric_percase(prediction == 2, label == 4))
    # metric_list.append(calculate_metric_percase(prediction == 6, label == 5))
    # metric_list.append(calculate_metric_percase(prediction == 10, label == 6))
    # metric_list.append(calculate_metric_percase(prediction == 1, label == 7))
    # metric_list.append(calculate_metric_percase(prediction == 7, label == 8))
    ##

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list