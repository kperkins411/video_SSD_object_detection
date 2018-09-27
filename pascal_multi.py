from fastai.conv_learner import *
from fastai.dataset import *

import json, pdb
from PIL import ImageDraw, ImageFont
from matplotlib import patches, patheffects
import utilities as ut
import ssd_model1 as ssd1

if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object

PATH = Path('../../../data/VOCdevkit/VOC2007')
JPEGS = 'JPEGImages'
IMG_PATH = PATH / JPEGS
MC_CSV = PATH / 'tmp/mc.csv'
CLAS_CSV = PATH / 'tmp/clas.csv'
MBB_CSV = PATH / 'tmp/mbb.csv'

IMAGES, ANNOTATIONS, CATEGORIES = ['images', 'annotations', 'categories']
FILE_NAME, ID, IMG_ID, CAT_ID, BBOX = 'file_name', 'id', 'image_id', 'category_id', 'bbox'

def load_data():
    '''
    preprocesses and loads all needed data
    :return:
    '''
    # available globaly
    global cats, trn_fns, trn_ids, trn_anno, mc, mcs, df, id2cat, val_mcs, trn_mcs
    trn_j = json.load((PATH / 'pascal_train2007.json').open())

    # dictionary with pascal VOC categories(about 20 items for pascalVOC)
    cats = dict((o[ID], o['name']) for o in trn_j[CATEGORIES])

    # id, filename like {12: '000012.jpg'...
    trn_fns = dict((o[ID], o[FILE_NAME]) for o in trn_j[IMAGES])

    # just the IDs
    trn_ids = [o[ID] for o in trn_j[IMAGES]]

    # default dict for each image ID, holds bounding box and class Ids for each image
    # 12  <class 'list'>: [(array([ 96, 155, 269, 350]), 7)]        #1 item
    # 17 <class 'list'>: [(array([ 61, 184, 198, 278]), 15), (array([ 77,  89, 335, 402]), 13)] #2 items
    trn_anno = ut.get_trn_anno(trn_j=trn_j, ano=ANNOTATIONS, bbox=BBOX, img_ID=IMG_ID, cat_ID=CAT_ID)

    mc = [[cats[p[1]] for p in trn_anno[o]] for o in trn_ids]
    mcs = [' '.join(str(p) for p in o) for o in mc]
    id2cat = list(cats.values())
    cat2id = {v: k for k, v in enumerate(id2cat)}
    mcs = np.array([np.array([cat2id[p] for p in o]) for o in mc]);

    val_idxs = get_cv_idxs(len(trn_fns))
    ((val_mcs, trn_mcs),) = split_by_idx(val_idxs, mcs)

    mbb = [np.concatenate([p[0] for p in trn_anno[o]]) for o in trn_ids]
    mbbs = [' '.join(str(p) for p in o) for o in mbb]

    # convert to a dataframe and then save
    # first the classes
    df = pd.DataFrame({'fn': [trn_fns[o] for o in trn_ids], 'clas': mcs}, columns=['fn', 'clas'])
    df.to_csv(MC_CSV, index=False)

    # then the bounding boxes
    df = pd.DataFrame({'fn': [trn_fns[o] for o in trn_ids], 'bbox': mbbs}, columns=['fn', 'bbox'])
    df.to_csv(MBB_CSV, index=False)

def classes_only(use_saved_model=True):
    '''
    just predictions printed out on images
    :return:
    '''
    global PRETRAINED_MODEL, IMG_SZ, BATCH_SIZE, tfms, md, learn, lr, lrs, y, x, fig, axes, i, ax, ima

    # load a model
    PRETRAINED_MODEL = resnet34
    IMG_SZ = 224
    BATCH_SIZE = 64

    tfms = tfms_from_model(PRETRAINED_MODEL, IMG_SZ, crop_type=CropType.NO)
    md = ImageClassifierData.from_csv(PATH, JPEGS, MC_CSV, tfms=tfms, bs=BATCH_SIZE)
    learn = ConvLearner.pretrained(PRETRAINED_MODEL, md)
    learn.opt_fn = optim.Adam

    if (use_saved_model == False):
        #find a learning rate
        lrf = learn.lr_find(1e-5, 100)
        learn.sched.plot(0)

        # based on plot above choose following LR
        lr = 2e-2

        # train for a bit
        learn.fit(lr, 1, cycle_len=3, use_clr=(32, 5))

        # choose differential learning rates
        lrs = np.array([lr / 100, lr / 10, lr])

        # freeze all but last 2 layers
        learn.freeze_to(-2)

        # find a better LR
        learn.lr_find(lrs / 1000)
        learn.sched.plot(0)

        # train some more
        learn.fit(lrs / 10, 1, cycle_len=5, use_clr=(32, 5))

        # save then load the model
        learn.save('mclas')

    learn.load('mclas')

    y = learn.predict()
    x, _ = next(iter(md.val_dl))
    x = to_np(x)
    fig, axes = plt.subplots(3, 4, figsize=(12, 8))
    for i, ax in enumerate(axes.flat):
        ima = md.val_ds.denorm(x)[i]
        ya = np.nonzero(y[i] > 0.4)[0]
        b = '\n'.join(md.classes[o] for o in ya)
        ax = ut.show_img(ima, ax=ax)
        draw_text(ax, (0, 0), b)
    plt.tight_layout()


def get_ImageDataObject():
    global PRETRAINED_MODEL, IMG_SZ, BATCH_SIZE, tfms, md, x, y, fig, axes, i, ax
    PRETRAINED_MODEL = resnet34
    IMG_SZ = 224
    BATCH_SIZE = 64
    aug_tfms = [RandomRotate(3, p=0.5, tfm_y=TfmType.COORD),
                RandomLighting(0.05, 0.05, tfm_y=TfmType.COORD),
                RandomFlip(tfm_y=TfmType.COORD)]
    tfms = tfms_from_model(PRETRAINED_MODEL, IMG_SZ, crop_type=CropType.NO, tfm_y=TfmType.COORD, aug_tfms=aug_tfms)

    # manages the data
    md = ImageClassifierData.from_csv(PATH, JPEGS, MBB_CSV, tfms=tfms, bs=BATCH_SIZE, continuous=True, num_workers=4)

    #dataset  that joins ds and y2
    class ConcatLblDataset(Dataset):
        def __init__(self, ds, y2):
            self.ds, self.y2 = ds, y2
            self.sz = ds.sz

        def __len__(self): return len(self.ds)

        def __getitem__(self, i):
            x, y = self.ds[i]
            return (x, (y, self.y2[i]))

    trn_ds2 = ConcatLblDataset(md.trn_ds, trn_mcs)
    val_ds2 = ConcatLblDataset(md.val_ds, val_mcs)

    # sub in above datasets
    md.trn_dl.dataset = trn_ds2
    md.val_dl.dataset = val_ds2
    return md

def show_groundTruthLabelsAndBBoxes():
    x, y = to_np(next(iter(md.val_dl)))
    x = md.val_ds.ds.denorm(x)
    x, y = to_np(next(iter(md.trn_dl)))
    x = md.trn_ds.ds.denorm(x)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for i, ax in enumerate(axes.flat):
           # show_ground_truth(ax, im, bbox, id2cat1,
        ut.show_ground_truth(ax, x[i], y[0][i],id2cat1 = id2cat,clas=y[1][i]  )
    plt.tight_layout()
    plt.show(block = True)
# ### Train

def hw2corners(ctr, hw):
    return torch.cat([ctr - hw / 2, ctr + hw / 2], dim=1)

def one_hot_embedding(labels, num_classes):
    return torch.eye(num_classes)[labels.data.cpu()]

class BCE_Loss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, pred, targ):
        t = one_hot_embedding(targ, self.num_classes + 1)
        t = V(t[:, :-1].contiguous())  # .cpu()
        x = pred[:, :-1]
        w = self.get_weight(x, t)
        return F.binary_cross_entropy_with_logits(x, t, w, size_average=False) / self.num_classes

    def get_weight(self, x, t): return None


# loss_f = BCE_Loss(len(id2cat))

def intersect(box_a, box_b):
    max_xy = torch.min(box_a[:, None, 2:], box_b[None, :, 2:])
    min_xy = torch.max(box_a[:, None, :2], box_b[None, :, :2])
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def box_sz(b):
    return ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))

def jaccard(box_a, box_b):
    inter = intersect(box_a, box_b)
    union = box_sz(box_a).unsqueeze(1) + box_sz(box_b).unsqueeze(0) - inter
    return inter / union

def get_y(bbox, clas):
    bbox = bbox.view(-1, 4) / IMG_SZ
    bb_keep = ((bbox[:, 2] - bbox[:, 0]) > 0).nonzero()[:, 0]
    return bbox[bb_keep], clas[bb_keep]


def actn_to_bb(actn, anchors):
    actn_bbs = torch.tanh(actn)
    actn_centers = (actn_bbs[:, :2] / 2 * grid_sizes) + anchors[:, :2]
    actn_hw = (actn_bbs[:, 2:] / 2 + 1) * anchors[:, 2:]
    return hw2corners(actn_centers, actn_hw)


def map_to_ground_truth(overlaps, print_it=False):
    prior_overlap, prior_idx = overlaps.max(1)
    if print_it: print(prior_overlap)
    #     pdb.set_trace()
    gt_overlap, gt_idx = overlaps.max(0)
    gt_overlap[prior_idx] = 1.99
    for i, o in enumerate(prior_idx): gt_idx[o] = i
    return gt_overlap, gt_idx


def ssd_1_loss(b_c, b_bb, bbox, clas, print_it=False):
    bbox, clas = get_y(bbox, clas)
    a_ic = actn_to_bb(b_bb, anchors)
    overlaps = jaccard(bbox.data, anchor_cnr.data)
    gt_overlap, gt_idx = map_to_ground_truth(overlaps, print_it)
    gt_clas = clas[gt_idx]
    pos = gt_overlap > 0.4
    pos_idx = torch.nonzero(pos)[:, 0]
    gt_clas[1 - pos] = len(id2cat)
    gt_bbox = bbox[gt_idx]
    loc_loss = ((a_ic[pos_idx] - gt_bbox[pos_idx]).abs()).mean()
    clas_loss = loss_f(b_c, gt_clas)
    return loc_loss, clas_loss


def ssd_loss(pred, targ, print_it=False):
    lcs, lls = 0., 0.
    for b_c, b_bb, bbox, clas in zip(*pred, *targ):
        loc_loss, clas_loss = ssd_1_loss(b_c, b_bb, bbox, clas, print_it)
        lls += loc_loss
        lcs += clas_loss
    if print_it: print(f'loc: {lls.data[0]}, clas: {lcs.data[0]}')
    return lls + lcs


def get_anchors(k):
    '''
    k:
    defines the number of anchors
    :return:
    '''
    global i, anchors, grid_sizes, anchor_cnr
    anc_grid = 4
    anc_offset = 1 / (anc_grid * 2)
    anc_x = np.repeat(np.linspace(anc_offset, 1 - anc_offset, anc_grid), anc_grid)
    anc_y = np.tile(np.linspace(anc_offset, 1 - anc_offset, anc_grid), anc_grid)
    anc_ctrs = np.tile(np.stack([anc_x, anc_y], axis=1), (k, 1))
    anc_sizes = np.array([[1 / anc_grid, 1 / anc_grid] for i in range(anc_grid * anc_grid)])
    anchors = V(np.concatenate([anc_ctrs, anc_sizes], axis=1), requires_grad=False).float()
    grid_sizes = V(np.array([1 / anc_grid]), requires_grad=False).unsqueeze(1)
    anchor_cnr = hw2corners(anchors[:, :2], anchors[:, 2:])


def unknown():
    global x, y, fig, ax, i, lr, lrs, ima
    x, y = next(iter(md.val_dl))
    # x,y = V(x).cpu(),V(y)
    x, y = V(x), V(y)
    fig, ax = plt.subplots(figsize=(7, 7))
    for i, o in enumerate(y): y[i] = o.cuda()
    learn.model.cuda()
    batch = learn.model(x)
    # uncomment to debug on cpu
    # anchors = anchors.cpu(); grid_sizes = grid_sizes.cpu(); anchor_cnr = anchor_cnr.cpu()
    ssd_loss(batch, y, True)
    learn.crit = ssd_loss
    lr = 3e-3
    lrs = np.array([lr / 100, lr / 10, lr])
    learn.lr_find(lrs / 1000, 1.)
    learn.sched.plot(1)
    learn.fit(lr, 1, cycle_len=5, use_clr=(20, 10))
    learn.save('0')
    learn.load('0')
    # ### Testing
    x, y = next(iter(md.val_dl))
    x, y = V(x), V(y)
    learn.model.eval()
    batch = learn.model(x)
    b_clas, b_bb = batch
    b_clas.size(), b_bb.size()
    idx = 7
    b_clasi = b_clas[idx]
    b_bboxi = b_bb[idx]
    ima = md.val_ds.ds.denorm(to_np(x))[idx]
    bbox, clas = get_y(y[0][idx], y[1][idx])
    bbox, clas

    def torch_gt(ax, ima, bbox, clas, prs=None, thresh=0.4):
        return show_ground_truth(ax, ima, to_np((bbox * 224).long()),
                                 to_np(clas), to_np(prs) if prs is not None else None, thresh)

    fig, ax = plt.subplots(figsize=(7, 7))
    torch_gt(ax, ima, bbox, clas)

#
#
#
# fig, ax = plt.subplots(figsize=(7, 7))
# torch_gt(ax, ima, anchor_cnr, b_clasi.max(1)[1])
#
#
#
# a_ic = actn_to_bb(b_bboxi, anchors)
#
#
#
#
# fig, ax = plt.subplots(figsize=(7, 7))
# torch_gt(ax, ima, a_ic, b_clasi.max(1)[1], b_clasi.max(1)[0].sigmoid(), thresh=0.0)
#
#
#
#
# overlaps = jaccard(bbox.data, anchor_cnr.data)
# overlaps
#
#
#
#
#
# gt_overlap, gt_idx = map_to_ground_truth(overlaps)
# gt_overlap, gt_idx
#
#
#
#
# gt_clas = clas[gt_idx];
# gt_clas
#
#
#
#
# thresh = 0.5
# pos = gt_overlap > thresh
# pos_idx = torch.nonzero(pos)[:, 0]
# neg_idx = torch.nonzero(1 - pos)[:, 0]
# pos_idx
#
#
#
#
# gt_clas[1 - pos] = len(id2cat)
# [id2cat[o] if o < len(id2cat) else 'bg' for o in gt_clas.data]
#
#
#
#
# gt_bbox = bbox[gt_idx]
# loc_loss = ((a_ic[pos_idx] - gt_bbox[pos_idx]).abs()).mean()
# clas_loss = F.cross_entropy(b_clasi, gt_clas)
# loc_loss, clas_loss
#
#
#
#
# fig, axes = plt.subplots(3, 4, figsize=(16, 12))
# for idx, ax in enumerate(axes.flat):
#     ima = md.val_ds.ds.denorm(to_np(x))[idx]
#     bbox, clas = get_y(y[0][idx], y[1][idx])
#     ima = md.val_ds.ds.denorm(to_np(x))[idx]
#     bbox, clas = get_y(bbox, clas);
#     bbox, clas
#     a_ic = actn_to_bb(b_bb[idx], anchors)
#     torch_gt(ax, ima, a_ic, b_clas[idx].max(1)[1], b_clas[idx].max(1)[0].sigmoid(), 0.01)
# plt.tight_layout()
#
# # ## More anchors!
#
# # ### Create anchors
#
#
#
#
# anc_grids = [4, 2, 1]
# # anc_grids = [2]
# anc_zooms = [0.7, 1., 1.3]
# # anc_zooms = [1.]
# anc_ratios = [(1., 1.), (1., 0.5), (0.5, 1.)]
# # anc_ratios = [(1.,1.)]
# anchor_scales = [(anz * i, anz * j) for anz in anc_zooms for (i, j) in anc_ratios]
# k = len(anchor_scales)
# anc_offsets = [1 / (o * 2) for o in anc_grids]
# k
#
#
#
#
# anc_x = np.concatenate([np.repeat(np.linspace(ao, 1 - ao, ag), ag)
#                         for ao, ag in zip(anc_offsets, anc_grids)])
# anc_y = np.concatenate([np.tile(np.linspace(ao, 1 - ao, ag), ag)
#                         for ao, ag in zip(anc_offsets, anc_grids)])
# anc_ctrs = np.repeat(np.stack([anc_x, anc_y], axis=1), k, axis=0)
#
#
#
#
# anc_sizes = np.concatenate([np.array([[o / ag, p / ag] for i in range(ag * ag) for o, p in anchor_scales])
#                             for ag in anc_grids])
# grid_sizes = V(np.concatenate([np.array([1 / ag for i in range(ag * ag) for o, p in anchor_scales])
#                                for ag in anc_grids]), requires_grad=False).unsqueeze(1)
# anchors = V(np.concatenate([anc_ctrs, anc_sizes], axis=1), requires_grad=False).float()
# anchor_cnr = hw2corners(anchors[:, :2], anchors[:, 2:])
#
#
#
#
# anchors
#
#
#
#
# x, y = to_np(next(iter(md.val_dl)))
# x = md.val_ds.ds.denorm(x)
#
# a = np.reshape((to_np(anchor_cnr) + to_np(torch.randn(*anchor_cnr.size())) * 0.01) * 224, -1)
#
# fig, ax = plt.subplots(figsize=(7, 7))
# ut.show_ground_truth(ax, x[0], a)
#
# fig, ax = plt.subplots(figsize=(7, 7))
# ut.show_ground_truth(ax, x[0], a)
#
# # ### Model
#
#
#
#
# drop = 0.4
#
#
# class SSD_MultiHead(nn.Module):
#     def __init__(self, k, bias):
#         super().__init__()
#         self.drop = nn.Dropout(drop)
#         self.sconv0 = StdConv(512, 256, stride=1, drop=drop)
#         self.sconv1 = StdConv(256, 256, drop=drop)
#         self.sconv2 = StdConv(256, 256, drop=drop)
#         self.sconv3 = StdConv(256, 256, drop=drop)
#         self.out0 = OutConv(k, 256, bias)
#         self.out1 = OutConv(k, 256, bias)
#         self.out2 = OutConv(k, 256, bias)
#         self.out3 = OutConv(k, 256, bias)
#
#     def forward(self, x):
#         x = self.drop(F.relu(x))
#         x = self.sconv0(x)
#         x = self.sconv1(x)
#         o1c, o1l = self.out1(x)
#         x = self.sconv2(x)
#         o2c, o2l = self.out2(x)
#         x = self.sconv3(x)
#         o3c, o3l = self.out3(x)
#         return [torch.cat([o1c, o2c, o3c], dim=1),
#                 torch.cat([o1l, o2l, o3l], dim=1)]
#
#
# head_reg4 = SSD_MultiHead(k, -4.)
# models = ConvnetBuilder(PRETRAINED_MODEL, 0, 0, 0, custom_head=head_reg4)
# learn = ConvLearner(md, models)
# learn.opt_fn = optim.Adam
#
# learn.crit = ssd_loss
# lr = 1e-2
# lrs = np.array([lr / 100, lr / 10, lr])
#
# x, y = next(iter(md.val_dl))
# x, y = V(x), V(y)
# batch = learn.model(V(x))
#
# batch[0].size(), batch[1].size()
#
# ssd_loss(batch, y, True)
#
# learn.lr_find(lrs / 1000, 1.)
# learn.sched.plot(n_skip_end=2)
#
# learn.fit(lrs, 1, cycle_len=4, use_clr=(20, 8))
#
# learn.save('tmp')
#
# learn.freeze_to(-2)
# learn.fit(lrs / 2, 1, cycle_len=4, use_clr=(20, 8))
#
# learn.save('prefocal')
#
#
#
#
# x, y = next(iter(md.val_dl))
# y = V(y)
# batch = learn.model(V(x))
# b_clas, b_bb = batch
# x = to_np(x)
#
# fig, axes = plt.subplots(3, 4, figsize=(16, 12))
# for idx, ax in enumerate(axes.flat):
#     ima = md.val_ds.ds.denorm(x)[idx]
#     bbox, clas = get_y(y[0][idx], y[1][idx])
#     a_ic = actn_to_bb(b_bb[idx], anchors)
#     torch_gt(ax, ima, a_ic, b_clas[idx].max(1)[1], b_clas[idx].max(1)[0].sigmoid(), 0.21)
# plt.tight_layout()
#
#
# # ## Focal loss
# def plot_results(thresh):
#     x, y = next(iter(md.val_dl))
#     y = V(y)
#     batch = learn.model(V(x))
#     b_clas, b_bb = batch
#
#     x = to_np(x)
#     fig, axes = plt.subplots(3, 4, figsize=(16, 12))
#     for idx, ax in enumerate(axes.flat):
#         ima = md.val_ds.ds.denorm(x)[idx]
#         bbox, clas = get_y(y[0][idx], y[1][idx])
#         a_ic = actn_to_bb(b_bb[idx], anchors)
#         clas_pr, clas_ids = b_clas[idx].max(1)
#         clas_pr = clas_pr.sigmoid()
#         torch_gt(ax, ima, a_ic, clas_ids, clas_pr, clas_pr.max().data[0] * thresh)
#     plt.tight_layout()
#
# class FocalLoss(BCE_Loss):
#     def get_weight(self, x, t):
#         alpha, gamma = 0.25, 1
#         p = x.sigmoid()
#         pt = p * t + (1 - p) * (1 - t)
#         w = alpha * t + (1 - alpha) * (1 - t)
#         return w * (1 - pt).pow(gamma)
#
# loss_f = FocalLoss(len(id2cat))
#
# x, y = next(iter(md.val_dl))
# x, y = V(x), V(y)
# batch = learn.model(x)
# ssd_loss(batch, y, True)
#
# learn.lr_find(lrs / 1000, 1.)
# learn.sched.plot(n_skip_end=1)
#
# learn.fit(lrs, 1, cycle_len=10, use_clr=(20, 10))
#
# learn.save('fl0')
# learn.load('fl0')
#
# learn.freeze_to(-2)
# learn.fit(lrs / 4, 1, cycle_len=10, use_clr=(20, 10))
#
# learn.save('drop4')
# learn.load('drop4')
# plot_results(0.75)
#
#
# # ## NMS
# def nms(boxes, scores, overlap=0.5, top_k=100):
#     keep = scores.new(scores.size(0)).zero_().long()
#     if boxes.numel() == 0: return keep
#     x1 = boxes[:, 0]
#     y1 = boxes[:, 1]
#     x2 = boxes[:, 2]
#     y2 = boxes[:, 3]
#     area = torch.mul(x2 - x1, y2 - y1)
#     v, idx = scores.sort(0)  # sort in ascending order
#     idx = idx[-top_k:]  # indices of the top-k largest vals
#     xx1 = boxes.new()
#     yy1 = boxes.new()
#     xx2 = boxes.new()
#     yy2 = boxes.new()
#     w = boxes.new()
#     h = boxes.new()
#
#     count = 0
#     while idx.numel() > 0:
#         i = idx[-1]  # index of current largest val
#         keep[count] = i
#         count += 1
#         if idx.size(0) == 1: break
#         idx = idx[:-1]  # remove kept element from view
#         # load bboxes of next highest vals
#         torch.index_select(x1, 0, idx, out=xx1)
#         torch.index_select(y1, 0, idx, out=yy1)
#         torch.index_select(x2, 0, idx, out=xx2)
#         torch.index_select(y2, 0, idx, out=yy2)
#         # store element-wise max with next highest score
#         xx1 = torch.clamp(xx1, min=x1[i])
#         yy1 = torch.clamp(yy1, min=y1[i])
#         xx2 = torch.clamp(xx2, max=x2[i])
#         yy2 = torch.clamp(yy2, max=y2[i])
#         w.resize_as_(xx2)
#         h.resize_as_(yy2)
#         w = xx2 - xx1
#         h = yy2 - yy1
#         # check sizes of xx1 and xx2.. after each iteration
#         w = torch.clamp(w, min=0.0)
#         h = torch.clamp(h, min=0.0)
#         inter = w * h
#         # IoU = i / (area(a) + area(b) - i)
#         rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
#         union = (rem_areas - inter) + area[i]
#         IoU = inter / union  # store result in iou
#         # keep only elements with an IoU <= overlap
#         idx = idx[IoU.le(overlap)]
#     return keep, count
#
#
#
#
#
# x, y = next(iter(md.val_dl))
# y = V(y)
# batch = learn.model(V(x))
# b_clas, b_bb = batch
# x = to_np(x)
#
#
#
#
#
# def show_nmf(idx):
#     ima = md.val_ds.ds.denorm(x)[idx]
#     bbox, clas = get_y(y[0][idx], y[1][idx])
#     a_ic = actn_to_bb(b_bb[idx], anchors)
#     clas_pr, clas_ids = b_clas[idx].max(1)
#     clas_pr = clas_pr.sigmoid()
#
#     conf_scores = b_clas[idx].sigmoid().t().data
#
#     out1, out2, cc = [], [], []
#     for cl in range(0, len(conf_scores) - 1):
#         c_mask = conf_scores[cl] > 0.25
#         if c_mask.sum() == 0: continue
#         scores = conf_scores[cl][c_mask]
#         l_mask = c_mask.unsqueeze(1).expand_as(a_ic)
#         boxes = a_ic[l_mask].view(-1, 4)
#         ids, count = nms(boxes.data, scores, 0.4, 50)
#         ids = ids[:count]
#         out1.append(scores[ids])
#         out2.append(boxes.data[ids])
#         cc.append([cl] * count)
#     if not cc:
#         print(f"{i}: empty array")
#         return
#     cc = T(np.concatenate(cc))
#     out1 = torch.cat(out1)
#     out2 = torch.cat(out2)
#
#     fig, ax = plt.subplots(figsize=(8, 8))
#     torch_gt(ax, ima, out2, cc, out1, 0.1)
#
# for i in range(12): show_nmf(i)


if __name__=="__main__":
    k=1

    load_data()
    # show images with just the classes in them after training
    # classes_only()

    #get the object that will feed us train, validate, test data for consumption
    md = get_ImageDataObject()

    #show both the image and the ground truth preds and bounding boxes
    show_groundTruthLabelsAndBBoxes()  # works

    #get a model
    learn = ssd1.get_model(id2cat, md,k)

    x, y = next(iter(md.val_dl))


    get_anchors(k)


