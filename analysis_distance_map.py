import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from matplotlib import cm

from util import html

cyclegan_path = 'results/brats_flair_t1_rawcyclegan_autoaddtumor/test_latest/images/'
distance_path = 'results/brats_flair_t1_3/test_latest/images/'
out_root = 'results/analysis_heatmap/'
out_path = 'results/analysis_heatmap/images/'

opt_cell_range = 16
opt_crop_size = 8
opt_threshold = 1.0
opt_hist_dim = 256 / opt_cell_range
opt_mu = np.array([[range(opt_hist_dim)]])

cmap = np.array(cm.jet(range(256)))
cmap = cmap[:,:3]
cmap = list(np.int32(cmap.flatten() *255))

def im2col(mat_in):
    crop_num = mat_in.shape[0] / opt_crop_size
    tmp = np.vstack(np.hsplit(mat_in, crop_num))
    tmp = np.stack(np.vsplit(tmp, crop_num*crop_num))
    tmp = tmp.reshape(tmp.shape[0], tmp.shape[1]*tmp.shape[2])
    return tmp

def softhist(mat_in):
    mat_tmp = np.expand_dims(mat_in, axis=-1)
    mat_tmp1 = np.repeat(mat_tmp, opt_hist_dim, axis=-1)
    tmp_mu = np.repeat(opt_mu, mat_tmp1.shape[0], axis=0)
    tmp_mu = np.repeat(tmp_mu, mat_tmp1.shape[1], axis=1)
    tmp = np.maximum(1 - np.abs(mat_tmp1 - tmp_mu), 0)
    tmp_sum = np.sum(tmp, axis=1)
    return tmp_sum

def gather_diff(mat_in):
    mat_tmp = np.expand_dims(mat_in, axis=1)
    mat_tmp1 = np.repeat(mat_tmp, mat_tmp.shape[0], axis=1)
    mat_tmp2 = np.transpose(mat_tmp1, (1,0,2))
    tmp_abs = np.abs(mat_tmp1 - mat_tmp2)
    return tmp_abs

def draw_heatmap(heatmap, res_path):
    mat = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()) * 255
    img = Image.fromarray(np.uint8(np.transpose(mat, (1,0))))
    img = img.resize([256, 256], Image.NEAREST)
    img.putpalette(cmap)
    img.save(res_path)

def get_heatmap(img_in, img_out, res_path):
    mat_in = np.array(img_in)[:,:,0] / opt_cell_range
    mat_out = np.array(img_out)[:,:,0] / opt_cell_range
    col_in = im2col(mat_in)
    col_out = im2col(mat_out)
    hist_fea_in = softhist(col_in)
    hist_fea_out = softhist(col_out)
    hist_diff_in = gather_diff(hist_fea_in)
    hist_diff_out = gather_diff(hist_fea_out)

    diff_in_mean = np.mean(hist_diff_in, axis=2)
    hist_diff_bool = diff_in_mean <= opt_threshold
    hist_diff_bool_t = np.expand_dims(hist_diff_bool, axis=-1)
    hist_diff_bool_t = np.repeat(hist_diff_bool_t, hist_diff_out.shape[2], axis=2)
    hist_diff_out_bool = hist_diff_out * hist_diff_bool_t
    hist_diff_mean = np.mean(hist_diff_out_bool, axis=2)
    hist_diff_sum = np.sum(hist_diff_mean, axis=1)
    hist_diff_num = np.sum(hist_diff_bool, axis=1)
    hist_heatmap = hist_diff_sum / hist_diff_num
    heatmap = hist_heatmap.reshape(mat_in.shape[0]/opt_crop_size, mat_in.shape[1]/opt_crop_size)
    draw_heatmap(heatmap, res_path)

def draw_webpage_gt(img_name, out_path):
    real_A = Image.open(distance_path + img_name + '_real_A.png')
    real_B = Image.open(distance_path + img_name + '_real_B.png')
    real_A.save(out_path + img_name + '_gt_real_A.png')
    real_B.save(out_path + img_name + '_gt_real_B.png')
    get_heatmap(real_A, real_B, out_path + img_name + '_gt_dis_AB.png')
    get_heatmap(real_B, real_A, out_path + img_name + '_gt_dis_BA.png')
    ims = [img_name + '_gt_real_A.png', img_name + '_gt_real_B.png',
           img_name + '_gt_dis_AB.png', img_name + '_gt_dis_BA.png']
    txts = ['gt_real_A.png', 'gt_real_B.png',
            'gt_dis_AB.png', 'gt_dis_BA.png']
    links = [img_name + '_gt_real_A.png', img_name + '_gt_real_B.png',
             img_name + '_gt_dis_AB.png', img_name + '_gt_dis_BA.png']
    return ims, txts, links


def draw_webpage_res(img_name, in_path, out_path, suffix):
    real_A = Image.open(in_path + img_name + '_real_A.png')
    fake_B = Image.open(in_path + img_name + '_fake_B.png')
    rec_A = Image.open(in_path + img_name + '_rec_A.png')
    real_B = Image.open(in_path + img_name + '_real_B.png')
    fake_A = Image.open(in_path + img_name + '_fake_A.png')
    rec_B = Image.open(in_path + img_name + '_rec_B.png')
    real_A.save(out_path + img_name + suffix + '_real_A.png')
    fake_B.save(out_path + img_name + suffix + '_fake_B.png')
    rec_A.save(out_path + img_name + suffix + '_rec_A.png')
    real_B.save(out_path + img_name + suffix + '_real_B.png')
    fake_A.save(out_path + img_name + suffix + '_fake_A.png')
    rec_B.save(out_path + img_name + suffix + '_rec_B.png')
    get_heatmap(real_A, fake_B, out_path + img_name + suffix + '_dis_AB.png')
    get_heatmap(fake_B, rec_A, out_path + img_name + suffix + '_dis_BArec.png')
    get_heatmap(real_B, fake_A, out_path + img_name + suffix + '_dis_BA.png')
    get_heatmap(fake_A, rec_B, out_path + img_name + suffix + '_dis_ABrec.png')
    ims = [img_name + suffix + '_real_A.png', img_name + suffix + '_fake_B.png', img_name + suffix + '_rec_A.png',
           img_name + suffix + '_dis_AB.png', img_name + suffix + '_dis_BArec.png',
           img_name + suffix + '_real_B.png', img_name + suffix + '_fake_A.png', img_name + suffix + '_rec_B.png',
           img_name + suffix + '_dis_BA.png', img_name + suffix + '_dis_ABrec.png']
    txts = [suffix + '_real_A', suffix + '_fake_B', suffix + '_rec_A',
            suffix + '_dis_AB.png', suffix + '_dis_BArec.png',
            suffix + '_real_B', suffix + '_fake_A', suffix + '_rec_B',
            suffix + '_dis_BA.png', suffix + '_dis_ABrec.png']
    links = [img_name + suffix + '_real_A.png', img_name + suffix + '_fake_B.png', img_name + suffix + '_rec_A.png',
             img_name + suffix + '_dis_AB.png', img_name + suffix + '_dis_BArec.png',
             img_name + suffix + '_real_B.png', img_name + suffix + '_fake_A.png', img_name + suffix + '_rec_B.png',
             img_name + suffix + '_dis_BA.png', img_name + suffix + '_dis_ABrec.png']
    return ims, txts, links

web_dir = out_root
webpage = html.HTML(web_dir, 'analysis_distance_heatmap')
for pre in ['HG', 'LG']:
    for idx in range(1, 25+1):
        img_name = '%s_%04d_090' %(pre, idx)
        print img_name
        webpage.add_header(img_name + '_groundtruth')
        ims, txts, links = draw_webpage_gt(img_name, out_path)
        webpage.add_images(ims, txts, links, width=256)
        ims, txts, links = draw_webpage_res(img_name, cyclegan_path, out_path, '_cyc')
        webpage.add_images(ims, txts, links, width=256)
        ims, txts, links = draw_webpage_res(img_name, distance_path, out_path, '_dis')
        webpage.add_images(ims, txts, links, width=256)
webpage.save()
