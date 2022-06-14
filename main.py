import tensorflow as tf
import os
import glob
import util
import models
import cv2
from os.path import join
import numpy as np
import time


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='watermark1', help="Experiment name")
    parser.add_argument('--secret_len', type=int, default=100, help="Watermark information length")
    parser.add_argument('--cover_h', type=int, default=400, help="Height of carrier image")
    parser.add_argument('--cover_w', type=int, default=400, help="Width of carrier image")
    parser.add_argument('--num_epochs', type=int, default=2, help="Epoch of train")
    parser.add_argument('--num_steps', type=int, default=140000, help="total steps")
    parser.add_argument('--batch_size', type=int, default=2, help="batch size")
    parser.add_argument('--lr', type=float, default=.0001, help="learning rate")
    parser.add_argument('--dataset_path', type=str, default="F:\\myproject\\Dataset\\test",
                        help="dataset path of train")
    parser.add_argument('--loss_lpips_ratio', type=float, default=1, help="ratio of lpips loss")
    parser.add_argument('--loss_lpips_step', type=int, default=15000)  #
    parser.add_argument('--loss_mse_ratio', type=float, default=1, help="ratio of mse loss")
    parser.add_argument('--loss_mse_step', type=int, default=15000)  #
    parser.add_argument('--loss_secret_ratio', type=float, default=1, help="ratio of secret loss")
    parser.add_argument('--loss_secret_step', type=int, default=1)  #

    parser.add_argument('--use_second', type=bool, default=False)  #
    parser.add_argument('--gauss_stddev', type=float, default=.02)  #
    parser.add_argument('--is_in_warp', type=bool, default=True)  #
    parser.add_argument('--max_warp', type=float, default=.1)  #
    parser.add_argument('--max_bri', type=float, default=.3)  #
    parser.add_argument('--rnd_sat', type=float, default=1.0)  #
    parser.add_argument('--max_hue', type=float, default=.1)  #
    parser.add_argument('--cts_low', type=float, default=.5)  #
    parser.add_argument('--cts_high', type=float, default=1.5)  #

    parser.add_argument('--warp_step', type=int, default=10000)  #
    parser.add_argument('--bri_step', type=int, default=1000)  #
    parser.add_argument('--sat_step', type=int, default=1000)  #
    parser.add_argument('--hue_step', type=int, default=1000)  #
    parser.add_argument('--gaussian_step', type=int, default=1000)  #
    parser.add_argument('--cts_step', type=int, default=1000)  #

    parser.add_argument('--only_secret_N', help="The first N steps only optimize secret loss", type=int, default=8000)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--start_step', type=int, default=0)
    parser.add_argument('--GPU', type=str, default='0')
    parser.add_argument('--attention_start', type=int, default=6000)
    parser.add_argument('--fast_step', type=int, default=7000)
    parser.add_argument('--restart', type=int, default=120000)
    parser.add_argument('--damping_end', type=float, default=0.25)
    parser.add_argument('--mse_gain', type=float, default=10.0)
    parser.add_argument('--mse_gain_epoch', type=int, default=20)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU

    dataset_path = args.dataset_path
    file_names = glob.glob(dataset_path + '/*')
    # print("#######################")
    # print(file_names)
    file_len = len(file_names)
    total_step = int(file_len / args.batch_size)

    # place_holder
    global_index_tensor = tf.Variable(0, trainable=False, name='global_step')
    TFM_pl = tf.placeholder(shape=[None, 2, 8], dtype=tf.float32, name="warp_matrix")
    loss_ratio_pl = tf.placeholder(shape=[3], dtype=tf.float32, name="loss_ratio")

    # build graph
    cover_batch, secret_batch = util.build_dataset(file_names, batch_size=args.batch_size, epoch=args.num_epochs + 1,
                                                   H=args.cover_h, W=args.cover_w,
                                                   secret_size=args.secret_len)
    Encoder = models.WatermarkEncoder(height=args.cover_h, width=args.cover_h, base_num=32)

    Deconder = models.WatermarkDecoder(secret_size=args.secret_len, height=args.cover_h, width=args.cover_w,
                                       base_num=32)

    loss_total, loss_secret, config_op, image_summary_op, bit_acc = models.make_graph3(Encoder, Deconder, cover_batch,
                                                                                      secret_batch, loss_ratio_pl, args,
                                                                                      TFM_pl, global_index_tensor)

    variables = tf.trainable_variables()
    total_optimizer = tf.train.AdamOptimizer(args.lr).minimize(loss_total, var_list=variables,
                                                               global_step=global_index_tensor)
    secret_loss_optimizer = tf.train.AdamOptimizer(args.lr).minimize(loss_secret, var_list=variables,
                                                                     global_step=global_index_tensor)

    secret_pl = tf.placeholder(shape=[None, args.secret_len], dtype=tf.float32, name="secret")
    cover_pl = tf.placeholder(shape=[None, args.cover_h, args.cover_w, 3], dtype=tf.float32, name="cover")
    watered_image, M_map, T_map, N_map = models.make_encode_graph(Encoder, cover_pl, secret_pl, args.damping_end)
    pre_secret = models.make_decode_graph(Deconder, cover_pl)

    saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=100, keep_checkpoint_every_n_hours=5)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    base_path = './run/' + args.exp_name
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    with open(base_path + '/' + 'config.txt', "a") as file:
        file.write("#####################################" + "\n")
        str_time = time.strftime("%Y.%m.%d %H:%M:%S", time.localtime())
        file.write("run time: " + str_time + '\n')
        for arg in vars(args):
            para = "{: <25} {: <25}".format(str(arg) + ':', str(getattr(args, arg))) + '\n'
            file.write(para)
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        writer = tf.summary.FileWriter(join(base_path, 'logs'), sess.graph)

        if args.pretrained is not None:
            saver.restore(sess, args.pretrained)
        if args.start_step != 0:
            sess.run(tf.assign(global_index_tensor, args.start_step))
        global_index = 0
        i = args.start_epoch
        train_epoch = args.num_epochs
        while i < train_epoch:
            i += 1
            for j in range(total_step):
                loss_mse_ratio = min(args.loss_mse_ratio * global_index / args.loss_mse_step, args.loss_mse_ratio)
                if i >= (train_epoch - args.mse_gain_epoch):
                    loss_mse_ratio *= args.mse_gain
                loss_lpips_ratio = min(args.loss_lpips_ratio * global_index / args.loss_lpips_step,
                                       args.loss_lpips_ratio)
                loss_secret_ratio = min(args.loss_secret_ratio * global_index / args.loss_secret_step,
                                        args.loss_secret_ratio)
                shift_ratio = min(args.max_warp * global_index / args.warp_step, args.max_warp)
                shift_ratio = np.random.uniform() * shift_ratio
                TFM = util.get_transform_matrix(args.cover_h, np.floor(args.cover_h * shift_ratio), args.batch_size)
                feed_dict = {TFM_pl: TFM, loss_ratio_pl: [loss_secret_ratio, loss_lpips_ratio, loss_mse_ratio]}
                if global_index < args.only_secret_N:
                    _, loss_np, global_index, bit_acc_np, config_np, loss_secret_np = sess.run(
                        [secret_loss_optimizer, loss_total, global_index_tensor, bit_acc, config_op, loss_secret],
                        feed_dict=feed_dict)
                else:
                    _, loss_np, global_index, bit_acc_np, config_np, loss_secret_np = sess.run(
                        [total_optimizer, loss_total, global_index_tensor, bit_acc, config_op, loss_secret], feed_dict=feed_dict)
                if global_index % 100 == 0:
                    print("###############################################################")
                    print("Epoch: {}    step: {}".format(i, j + 1))
                    print("total loss:{:.5f}      bit_acc:{:.5f}       secret loss:{:.5f}".format(loss_np, bit_acc_np, loss_secret_np))
                if global_index % 200 == 0:
                    writer.add_summary(config_np, global_index)
                    warp_scale = tf.Summary(
                        value=[tf.Summary.Value(tag='nosie_config/warp_shift_ratio', simple_value=shift_ratio)])
                    writer.add_summary(warp_scale, global_index)
                if global_index % 1000 == 0:
                    image_summary, global_index = sess.run([image_summary_op, global_index_tensor], feed_dict)
                    writer.add_summary(image_summary, global_index)
                if global_index % 20000 == 0:
                    saver.save(sess, join(base_path, 'checkpoints/') + args.exp_name + ".chkp",
                               global_step=global_index)

        tf.saved_model.simple_save(sess,
                                   join(base_path, 'model_save') + 'model' + time.strftime("%Y%m%d_%H%M%S",
                                                                                           time.localtime()),
                                   inputs={'secret': secret_pl, 'image': cover_pl},
                                   outputs={'watermarked_image': watered_image, 'residual': N_map,
                                            'predict_secret': pre_secret})
        coord.request_stop()
        coord.join(threads)
    writer.close()


if __name__ == '__main__':
    main()
