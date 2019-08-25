#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" one example to work with image retrieval
    Author: Qing Cheng & Nan Yang
    08.2019
"""

from ArtiRetrieval import ArtiRetrieval
import utils
import os
import cv2
import numpy as np
from config import Config
import pdb
import time
import csv


def main():
    query_folder = ''
    cfg = Config(query_folder)

    # download checkpoint and check it
    utils.download_ckpt()
    utils.check_ckpt()

    ckpt_path = utils.default_ckpt_path()
    batch_size = 4
    gpu_id = '1'
    gpu_fraction = 0.25
    test_obj = ArtiRetrieval(ckpt_path, batch_size, gpu_id, gpu_fraction)

    # generate the base of descriptors of a given directory
    des_base, base_img_paths = test_obj.generate_directory_descriptors(cfg.base_img_dir)

    # save the descriptors to the disk
    test_obj.save_descriptors(des_base, cfg.des_base_path)

    # load base descriptors
    des_base = test_obj.load_descriptors(cfg.des_base_path)

    # inference
    query_descriptors, query_img_paths = test_obj.generate_directory_descriptors(cfg.query_img_dir)

    fig_save_dir = os.path.join(os.path.dirname(cfg.query_img_dir), 'res')
    if not os.path.exists(fig_save_dir):
        os.mkdir(fig_save_dir)
    print('the plot will be saved in          ', fig_save_dir)

    # query and visualization
    for idx in range(query_descriptors.shape[0]):
        query_des = query_descriptors[idx]
        query_img_path = query_img_paths[idx]
        retrieved_idxes, retrieved_values = test_obj.get_topk_matches(query_des, des_base, topk=1)
        retrieved_img_path = base_img_paths[retrieved_idxes[0]]
        retrieved_value = retrieved_values[0]
        utils.vis_query_best_res(query_img_path, retrieved_value, retrieved_img_path, fig_save_dir,
                                 show_plot=False, vertical=True)

    print('visualization results are saved in ', fig_save_dir)

    # make a video demo
    utils.img2video(fig_save_dir, fps=10.0, output=fig_save_dir + '.mp4')


def generate_des_for_24dataset():
    query_folder = ''
    cfg = Config(query_folder)

    # download checkpoint and check it
    utils.download_ckpt()
    utils.check_ckpt()

    ckpt_path = utils.default_ckpt_path()
    batch_size = 8
    gpu_id = '1'
    gpu_fraction = 0.75
    arti_obj = ArtiRetrieval(ckpt_path, batch_size, gpu_id, gpu_fraction)

    base_dir = '/data/qing/datasets/24h_Dataset/'
    descriptor_base_dir = '/data/qing/datasets/24h_Dataset/ref_des_base/'

    seqs = ['2019-08-05_15-09-41', '2019-08-05_15-21-20', '2019-08-06_11-36-14', '2019-08-06_11-50-46']
    cams = ['cam0', 'cam1']
    img_types = ['kf', 'all']

    for seq in seqs:
        for cam in cams:
            img_dir = '{}/{}/undistorted_images/{}'.format(base_dir, seq, cam)
            des_base_path = '{}/desp_{}_all_{}.npy'.format(descriptor_base_dir, seq, cam)

            # generate the base of descriptors of a given directory
            des_base, base_img_paths = arti_obj.generate_directory_descriptors(img_dir)

            arti_obj.save_descriptors(des_base, des_base_path)

            kf_img_dir = '{}/{}/undistorted_images/kf/{}'.format(base_dir, seq, cam)
            des_base_path = '{}/desp_{}_kf_{}.npy'.format(descriptor_base_dir, seq, cam)

            # generate the base of descriptors of a given directory
            des_base, base_img_paths = arti_obj.generate_directory_descriptors(kf_img_dir)

            arti_obj.save_descriptors(des_base, des_base_path)

            print('finished ', img_dir)


# @profile
def cal_acc_for_24dataset():
    ckpt_path = utils.default_ckpt_path()
    batch_size = 8
    gpu_id = '1'
    gpu_fraction = 0.75
    arti_obj = ArtiRetrieval(ckpt_path, batch_size, gpu_id, gpu_fraction)

    base_dir = '/data/qing/datasets/24h_Dataset/'
    descriptor_base_dir = '/data/qing/datasets/24h_Dataset/ref_des_base/'

    seqs = ['2019-08-05_15-09-41', '2019-08-05_15-21-20', '2019-08-06_11-36-14', '2019-08-06_11-50-46']
    cams = ['cam0', 'cam1']
    img_types = ['kf', 'all']

    csv_lines = []
    topk = 10
    max_dis = 3

    for base_seq in seqs:
        print('\n**************************************'
              '\nCurrent base sequence: ', base_seq)

        img_dir = '{}/{}/undistorted_images/{}'.format(base_dir, base_seq, cams[0])
        gps_dir = '{}/{}/GPS_data_IR'.format(base_dir, base_seq)
        des_base_path = '{}/desp_{}_{}_{}.npy'.format(descriptor_base_dir, base_seq, img_types[0], cams[0])

        des_base = np.load(des_base_path)
        ts_base, gps_wgs_base = arti_obj.load_gps_poses(os.path.join(gps_dir, 'GPS_WGS84.csv'))

        for candi_seq in seqs:
            print('Processing querying in sequence: ', candi_seq)
            if candi_seq == base_seq:
                continue

            des_candi = np.load('{}/desp_{}_{}_{}.npy'.format(descriptor_base_dir, candi_seq, img_types[0], cams[0]))
            ts_candi, gps_wgs_gt = arti_obj.load_gps_poses('{}/{}/GPS_data_IR/GPS_WGS84.csv'.format(base_dir, candi_seq))

            queried_gps_l2 = []
            ts_l2 = []
            queried_gps_closest = []
            ts_closest = []
            distance = []

            # pure best L2
            st = time.time()
            for i, des in enumerate(des_candi):
                retrieved_idx, _ = arti_obj.get_topk_matches(des, des_base, topk=1)
                ts = ts_base[retrieved_idx]
                ts_l2.append(ts)
                gps = gps_wgs_base[retrieved_idx, :]
                queried_gps_l2.append(gps)

                # if i == 10:
                #     break

            print('L2 speed: {:2.2f} FPS'.format(len(queried_gps_l2) / (time.time() - st)))

            # # naive imp with two given gps
            # for i, des in enumerate(des_candi):
            #     retrieved_idx, _ = arti_obj.get_topk_matches(des, des_base, topk=topk)
            #     gps_candi = gps_wgs_base[retrieved_idx, :]
            #
            #     if len(queried_gps) > 1:
            #         pre2_gps = gps_wgs_gt[[i-2, i-1], :]
            #
            #         _, keep_gps_pose = arti_obj.refine_gps_based_on_distance(pre2_gps, gps_candi, max_dis=max_dis)
            #
            #     else:
            #         keep_gps_pose = gps_candi[0, :]
            #
            #     queried_gps.append(keep_gps_pose)


            # closest distance to the given gps
            st = time.time()
            for i, des in enumerate(des_candi):
                retrieved_idx, _ = arti_obj.get_topk_matches(des, des_base, topk=topk)
                gps_candi = gps_wgs_base[retrieved_idx, :]

                closest_idx, closest_dis, keep_gps_pose = arti_obj.refine_gps_closest(gps_wgs_gt[i, :], gps_candi)

                ts = ts_base[retrieved_idx][closest_idx]
                ts_closest.append(ts)

                queried_gps_closest.append(keep_gps_pose)
                distance.append(closest_dis)

                # if i == 10:
                #     break

            print('closest speed: {:2.2f} FPS'.format(len(queried_gps_closest) / (time.time() - st)))

            queried_gps_l2 = np.array(queried_gps_l2)
            queried_gps_l2 = np.squeeze(queried_gps_l2)

            queried_gps_closest = np.array(queried_gps_closest)
            queried_gps_closest = np.squeeze(queried_gps_closest)

            with open('{}/v2/{}_queried_gps_l2.csv'.format(gps_dir, candi_seq), 'w') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(['timestamp, longitude, latitude, altitude'])
                for i, ts in enumerate(ts_l2):
                    line_list = [str(ts[0]), str(queried_gps_l2[i, 0]), str(queried_gps_l2[i, 1]), str(queried_gps_l2[i, 2])]
                    writer.writerow(line_list)

            with open('{}/v2/{}_queried_gps_closest.csv'.format(gps_dir, candi_seq), 'w') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(['timestamp, longitude, latitude, altitude'])
                for i, ts in enumerate(ts_l2):
                    line_list = [str(ts[0]), str(queried_gps_closest[i, 0]), str(queried_gps_closest[i, 1]), str(queried_gps_closest[i, 2])]
                    writer.writerow(line_list)

            # np.savetxt('{}/{}_queried_gps_l2.csv'.format(gps_dir, candi_seq),
            #            np.append(ts_l2, queried_gps_l2, axis=1), delimiter=',')
            # np.savetxt('{}/{}_queried_gps_closest.csv'.format(gps_dir, candi_seq),
            #            np.append(ts_closest, queried_gps_closest, axis=1), delimiter=',')

            # distance = np.array(distance)
            # print('Percentage of dis < 2m: {:0.4f}'.format(np.sum(distance <= 2) / distance.shape[0]))

            accuracies = []

            for queried_gps in [queried_gps_l2, queried_gps_closest]:
                csv_line = [base_seq, candi_seq]

                acc_time = []

                for the in [1, 2, 3, 4, 5, 10, 15, 20]:
                    st = time.time()
                    acc = arti_obj.calculate_gps_acc(queried_gps, gps_wgs_gt[:len(queried_gps), :], threshold=the)
                    acc_time.append(time.time() - st)
                    accuracies.append(acc)
                    csv_line.append(acc)

                print('Accuracy calculation speed: {:2.2f} FPS'.format(len(queried_gps) / np.mean(np.array(acc_time))))
                csv_line = np.array(csv_line)
                csv_lines.append(csv_line)


        break

    csv_lines = np.array(csv_lines)
    np.savetxt('./qing/results/eval_temp_{}_{}_4Re2v2.csv'.format(topk, max_dis), csv_lines, fmt='%s', delimiter=',')


if __name__ == '__main__':
    cal_acc_for_24dataset()
