import os
import sys
import torch
import h5py
import json
from PIL import Image
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import random

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou

BOX_SCALE = 1024  # Scale at which we have the boxes


def _load_style_annotation_map(style_annotation_file):
    """
    Load mapping produced by tools/fashion_style_adapter.py (full or compact mapping file).

    Returns:
        dict[str, int] image_id -> style_label, or None if path is empty / missing.
    """
    if not style_annotation_file:
        return None
    path = os.path.expanduser(style_annotation_file)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    if isinstance(data, dict) and "by_image_id" in data:
        for k, v in data["by_image_id"].items():
            lab = v.get("style_label", -1) if isinstance(v, dict) else v
            out[str(k)] = int(lab)
        return out
    if isinstance(data, dict) and "images" in data:
        for _, v in data["images"].items():
            if not isinstance(v, dict):
                continue
            iid = v.get("image_id")
            if iid is None:
                continue
            st = v.get("style") or {}
            lab = st.get("style_label", -1)
            out[str(iid)] = int(lab)
        return out
    return None


def _load_style_train_sample_weights(style_annotation_file):
    """
    Optional per-image sampling weights from fashion_style_adapter (balance-aware mapping).
    Returns dict[str, float] image_id -> weight, or None.
    """
    if not style_annotation_file:
        return None
    path = os.path.expanduser(style_annotation_file)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return None
    raw = data.get("train_sample_weight_by_image_id")
    if not isinstance(raw, dict) or not raw:
        return None
    out = {}
    for k, v in raw.items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out if out else None


def _load_style_stratified_split_ids(split_file: str, split_name: str):
    """
    Load image_id set for train/val/test from tools/stratified_style_split.py output.
    Returns set of str(image_id), or None if disabled / missing.
    """
    if not split_file or not split_name:
        return None
    path = os.path.expanduser(split_file)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ids = data.get(split_name)
    if not isinstance(ids, list):
        return None
    out = set()
    for x in ids:
        if isinstance(x, int):
            out.add(str(x))
        else:
            s = str(x).strip()
            if s.isdigit():
                out.add(s)
            else:
                out.add(s)
    return out if out else None


class VGDataset(torch.utils.data.Dataset):

    def __init__(self, split, img_dir, roidb_file, dict_file, image_file, transforms=None,
                filter_empty_rels=True, num_im=-1, num_val_im=5000,
                filter_duplicate_rels=True, filter_non_overlap=True, flip_aug=False, custom_eval=False, custom_path='',
                predicate_set: str = "vg", style_annotation_file: str = "",
                style_stratified_split_file: str = "", style_stratified_split_strict: bool = True):
        """
        Torch dataset for VisualGenome
        Parameters:
            split: Must be train, test, or val
            img_dir: folder containing all vg images
            roidb_file:  HDF5 containing the GT boxes, classes, and relationships
            dict_file: JSON Contains mapping of classes/relationships to words
            image_file: HDF5 containing image filenames
            filter_empty_rels: True if we filter out images without relationships between
                             boxes. One might want to set this to false if training a detector.
            filter_duplicate_rels: Whenever we see a duplicate relationship we'll sample instead
            num_im: Number of images in the entire dataset. -1 for all images.
            num_val_im: Number of images in the validation set (must be less than num_im
               unless num_im is -1.)
        """
        # for debug
        # num_im = 10000
        # num_val_im = 4

        assert split in {'train', 'val', 'test'}
        self.flip_aug = flip_aug
        self.split = split
        self.img_dir = img_dir
        self.dict_file = dict_file
        self.roidb_file = roidb_file
        self.image_file = image_file
        self.filter_non_overlap = filter_non_overlap and self.split == 'train'
        self.filter_duplicate_rels = filter_duplicate_rels and self.split == 'train'
        self.transforms = transforms

        self.predicate_set = predicate_set
        self.ind_to_classes, self.ind_to_predicates, self.ind_to_attributes = load_info(
            dict_file,
            predicate_set=self.predicate_set,
        ) # contiguous 151, 51 containing __background__
        self.categories = {i : self.ind_to_classes[i] for i in range(len(self.ind_to_classes))}

        # image_id (str) -> int label in [0, num_style_classes), or -1 if unknown / missing
        self._style_by_image_id = _load_style_annotation_map(style_annotation_file)
        self._style_train_sample_weight_by_image_id = _load_style_train_sample_weights(style_annotation_file)
        self._style_stratified_ids = _load_style_stratified_split_ids(style_stratified_split_file, split)
        self._style_stratified_split_strict = bool(style_stratified_split_strict)

        self.custom_eval = custom_eval
        if self.custom_eval:
            self.get_custom_imgs(custom_path)
        else:
            use_strict_json = (
                self._style_stratified_ids is not None and self._style_stratified_split_strict
            )
            if use_strict_json:
                # image_file row order must match HDF5 row order (same as load_image_filenames).
                self.filenames, self.img_info = load_image_filenames(img_dir, image_file)
                id_per_h5_row = [str(x["image_id"]) for x in self.img_info]
                self.split_mask, self.gt_boxes, self.gt_classes, self.gt_attributes, self.relationships = load_graphs(
                    self.roidb_file,
                    self.split,
                    num_im,
                    num_val_im=num_val_im,
                    filter_empty_rels=filter_empty_rels,
                    filter_non_overlap=self.filter_non_overlap,
                    skip_h5_train_val_carve=False,
                    strict_stratified_ids=self._style_stratified_ids,
                    id_per_h5_row=id_per_h5_row,
                )
                self.filenames = [self.filenames[i] for i in np.where(self.split_mask)[0]]
                self.img_info = [self.img_info[i] for i in np.where(self.split_mask)[0]]
            else:
                skip_h5_tv = self._style_stratified_ids is not None and self.split in {"train", "val"}
                self.split_mask, self.gt_boxes, self.gt_classes, self.gt_attributes, self.relationships = load_graphs(
                    self.roidb_file,
                    self.split,
                    num_im,
                    num_val_im=num_val_im,
                    filter_empty_rels=filter_empty_rels,
                    filter_non_overlap=self.filter_non_overlap,
                    skip_h5_train_val_carve=skip_h5_tv,
                )

                self.filenames, self.img_info = load_image_filenames(img_dir, image_file)
                self.filenames = [self.filenames[i] for i in np.where(self.split_mask)[0]]
                self.img_info = [self.img_info[i] for i in np.where(self.split_mask)[0]]

                if self._style_stratified_ids is not None:
                    self._apply_style_stratified_split_filter()


    def __getitem__(self, index):
        #if self.split == 'train':
        #    while(random.random() > self.img_info[index]['anti_prop']):
        #        index = int(random.random() * len(self.filenames))
        if self.custom_eval:
            img = Image.open(self.custom_files[index]).convert("RGB")
            target = torch.LongTensor([-1])
            if self.transforms is not None:
                img, target = self.transforms(img, target)
            return img, target, index
        
        img = Image.open(self.filenames[index]).convert("RGB")
        if img.size[0] != self.img_info[index]['width'] or img.size[1] != self.img_info[index]['height']:
            print('='*20, ' ERROR index ', str(index), ' ', str(img.size), ' ', str(self.img_info[index]['width']), ' ', str(self.img_info[index]['height']), ' ', '='*20)

        flip_img = (random.random() > 0.5) and self.flip_aug and (self.split == 'train')
        
        target = self.get_groundtruth(index, flip_img)

        if flip_img:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, index


    def get_statistics(self):
        fg_matrix, bg_matrix = get_VG_statistics(
            img_dir=self.img_dir,
            roidb_file=self.roidb_file,
            dict_file=self.dict_file,
            image_file=self.image_file,
            must_overlap=True,
            predicate_set=self.predicate_set,
        )
        eps = 1e-3
        bg_matrix += 1
        fg_matrix[:, :, 0] = bg_matrix
        pred_dist = np.log(fg_matrix / fg_matrix.sum(2)[:, :, None] + eps)

        result = {
            'fg_matrix': torch.from_numpy(fg_matrix),
            'pred_dist': torch.from_numpy(pred_dist).float(),
            'obj_classes': self.ind_to_classes,
            'rel_classes': self.ind_to_predicates,
            'att_classes': self.ind_to_attributes,
        }
        return result

    def get_custom_imgs(self, path):
        self.custom_files = []
        self.img_info = []
        if os.path.isdir(path):
            for file_name in tqdm(os.listdir(path)):
                self.custom_files.append(os.path.join(path, file_name))
                img = Image.open(os.path.join(path, file_name)).convert("RGB")
                self.img_info.append({'width':int(img.width), 'height':int(img.height)})
        # Expecting a list of paths in a json file
        if os.path.isfile(path):
            file_list = json.load(open(path))
            for file in tqdm(file_list):
                self.custom_files.append(file)
                img = Image.open(file).convert("RGB")
                self.img_info.append({'width': int(img.width), 'height': int(img.height)})

    def get_img_info(self, index):
        # WARNING: original image_file.json has several pictures with false image size
        # use correct function to check the validity before training
        # it will take a while, you only need to do it once

        # correct_img_info(self.img_dir, self.image_file)
        return self.img_info[index]

    def get_style_train_sample_weight(self, index):
        """For WeightedRandomSampler: >1 oversamples rare style classes (train split only)."""
        if self.split != "train" or not self._style_train_sample_weight_by_image_id:
            return 1.0
        if self.custom_eval or index >= len(self.img_info):
            return 1.0
        img_id = self.img_info[index].get("image_id", None)
        if img_id is None:
            try:
                base = os.path.basename(self.filenames[index])
                img_id = int(os.path.splitext(base)[0])
            except Exception:
                return 1.0
        w = self._style_train_sample_weight_by_image_id.get(str(img_id), 1.0)
        return float(w) if w > 0 else 1.0

    def _apply_style_stratified_split_filter(self):
        """Keep only images whose image_id is listed in fashion_style_splits.json for this split."""
        allowed = self._style_stratified_ids
        keep = []
        for j in range(len(self.filenames)):
            iid = self.img_info[j].get("image_id", None)
            if iid is None:
                try:
                    base = os.path.basename(self.filenames[j])
                    iid = int(os.path.splitext(base)[0])
                except Exception:
                    continue
            if str(iid) in allowed:
                keep.append(j)
        if not keep:
            raise RuntimeError(
                "STYLE_STRATIFIED_SPLIT_FILE: no images left after filtering for split=%r (check id format vs JSON)." % self.split
            )
        self.gt_boxes = [self.gt_boxes[j] for j in keep]
        self.gt_classes = [self.gt_classes[j] for j in keep]
        self.gt_attributes = [self.gt_attributes[j] for j in keep]
        self.relationships = [self.relationships[j] for j in keep]
        self.filenames = [self.filenames[j] for j in keep]
        self.img_info = [self.img_info[j] for j in keep]

    def get_groundtruth(self, index, evaluation=False, flip_img=False):
        img_info = self.get_img_info(index)
        w, h = img_info['width'], img_info['height']
        # important: recover original box from BOX_SCALE
        box = self.gt_boxes[index] / BOX_SCALE * max(w, h)
        box = torch.from_numpy(box).reshape(-1, 4)  # guard against no boxes
        if flip_img:
            new_xmin = w - box[:,2]
            new_xmax = w - box[:,0]
            box[:,0] = new_xmin
            box[:,2] = new_xmax
        target = BoxList(box, (w, h), 'xyxy') # xyxy

        target.add_field("labels", torch.from_numpy(self.gt_classes[index]))
        target.add_field("attributes", torch.from_numpy(self.gt_attributes[index]))

        relation = self.relationships[index].copy() # (num_rel, 3)
        if self.filter_duplicate_rels:
            # Filter out dupes!
            assert self.split == 'train'
            old_size = relation.shape[0]
            all_rel_sets = defaultdict(list)
            for (o0, o1, r) in relation:
                all_rel_sets[(o0, o1)].append(r)
            relation = [(k[0], k[1], np.random.choice(v)) for k,v in all_rel_sets.items()]
            relation = np.array(relation, dtype=np.int32)
        # HDF5 stores Visual Genome predicate ids; map to fashion (or other) ontology so
        # relation_map / relation_tuple match NUM_CLASSES and SGG eval (same as get_VG_statistics).
        ps = (self.predicate_set or "").strip().lower()
        if ps not in {"vg", "visual_genome", "visual-genome"}:
            for i in range(relation.shape[0]):
                relation[i, 2] = _map_vg_rel_index_to_fashion(
                    int(relation[i, 2]), self.dict_file, self.predicate_set
                )
        
        # add relation to target
        num_box = len(target)
        relation_map = torch.zeros((num_box, num_box), dtype=torch.int64)
        for i in range(relation.shape[0]):
            if relation_map[int(relation[i,0]), int(relation[i,1])] > 0:
                if (random.random() > 0.5):
                    relation_map[int(relation[i,0]), int(relation[i,1])] = int(relation[i,2])
            else:
                relation_map[int(relation[i,0]), int(relation[i,1])] = int(relation[i,2])
        target.add_field("relation", relation_map, is_triplet=True)

        # Image-level fashion style (for style head); -1 = ignore in style_loss (CrossEntropy ignore_index)
        style_label = -1
        if self._style_by_image_id is not None and len(self.filenames) > index:
            img_id = None
            if index < len(self.img_info):
                img_id = self.img_info[index].get("image_id", None)
            if img_id is None:
                try:
                    base = os.path.basename(self.filenames[index])
                    img_id = int(os.path.splitext(base)[0])
                except Exception:
                    img_id = None
            if img_id is not None:
                style_label = int(self._style_by_image_id.get(str(img_id), -1))
        # Image-level field: must use is_image_level=True so BoxList.clip_to_image / __getitem__ do not index it per-box
        target.add_field(
            "style_label",
            torch.tensor(style_label, dtype=torch.long),
            is_image_level=True,
        )

        if evaluation:
            target = target.clip_to_image(remove_empty=False)
            target.add_field("relation_tuple", torch.LongTensor(relation)) # for evaluation
            return target
        target = target.clip_to_image(remove_empty=True)
        return target

    def __len__(self):
        if self.custom_eval:
            return len(self.custom_files)
        return len(self.filenames)


def _map_vg_rel_index_to_fashion(gtr: int, dict_file: str, fashion_predicate_set: str) -> int:
    """H5 stores Visual Genome predicate indices; map by name into fashion_context (or other) ontology."""
    _, vg_predicates, _ = load_info(dict_file, predicate_set="vg")
    from maskrcnn_benchmark.data.datasets.fashion_predicates import get_predicate_set
    fashion_predicates = get_predicate_set(fashion_predicate_set)
    ft = {p: i for i, p in enumerate(fashion_predicates)}
    gtr = int(gtr)
    if gtr < 0 or gtr >= len(vg_predicates):
        return 0
    name = vg_predicates[gtr]
    return int(ft.get(name, 0))


def get_VG_statistics(
    img_dir,
    roidb_file,
    dict_file,
    image_file,
    must_overlap=True,
    predicate_set: str = "vg",
):
    # Statistics pass: no style JSON needed (faster); predicate_set must match training ontology size.
    train_data = VGDataset(
        split="train",
        img_dir=img_dir,
        roidb_file=roidb_file,
        dict_file=dict_file,
        image_file=image_file,
        num_val_im=5000,
        filter_duplicate_rels=False,
        predicate_set=predicate_set,
        style_annotation_file="",
    )
    num_obj_classes = len(train_data.ind_to_classes)
    num_rel_classes = len(train_data.ind_to_predicates)
    fg_matrix = np.zeros((num_obj_classes, num_obj_classes, num_rel_classes), dtype=np.int64)
    bg_matrix = np.zeros((num_obj_classes, num_obj_classes), dtype=np.int64)

    ps = (predicate_set or "vg").strip().lower()
    use_fashion_map = ps not in {"vg", "visual_genome", "visual-genome"}

    for ex_ind in tqdm(range(len(train_data))):
        gt_classes = train_data.gt_classes[ex_ind].copy()
        gt_relations = train_data.relationships[ex_ind].copy()
        gt_boxes = train_data.gt_boxes[ex_ind].copy()

        # For the foreground, we'll just look at everything
        o1o2 = gt_classes[gt_relations[:, :2]]
        for (o1, o2), gtr in zip(o1o2, gt_relations[:, 2]):
            if use_fashion_map:
                gti = _map_vg_rel_index_to_fashion(int(gtr), dict_file, predicate_set)
            else:
                gti = int(gtr)
            if 0 <= gti < num_rel_classes:
                fg_matrix[o1, o2, gti] += 1
        # For the background, get all of the things that overlap.
        o1o2_total = gt_classes[np.array(
            box_filter(gt_boxes, must_overlap=must_overlap), dtype=int)]
        for (o1, o2) in o1o2_total:
            bg_matrix[o1, o2] += 1

    return fg_matrix, bg_matrix
    

def box_filter(boxes, must_overlap=False):
    """ Only include boxes that overlap as possible relations. 
    If no overlapping boxes, use all of them."""
    n_cands = boxes.shape[0]

    overlaps = bbox_overlaps(boxes.astype(np.float32), boxes.astype(np.float32), to_move=0) > 0
    np.fill_diagonal(overlaps, 0)

    all_possib = np.ones_like(overlaps, dtype=bool)
    np.fill_diagonal(all_possib, 0)

    if must_overlap:
        possible_boxes = np.column_stack(np.where(overlaps))

        if possible_boxes.size == 0:
            possible_boxes = np.column_stack(np.where(all_possib))
    else:
        possible_boxes = np.column_stack(np.where(all_possib))
    return possible_boxes

def bbox_overlaps(boxes1, boxes2, to_move=1):
    """
    boxes1 : numpy, [num_obj, 4] (x1,y1,x2,y2)
    boxes2 : numpy, [num_obj, 4] (x1,y1,x2,y2)
    """
    #print('boxes1: ', boxes1.shape)
    #print('boxes2: ', boxes2.shape)
    num_box1 = boxes1.shape[0]
    num_box2 = boxes2.shape[0]
    lt = np.maximum(boxes1.reshape([num_box1, 1, -1])[:,:,:2], boxes2.reshape([1, num_box2, -1])[:,:,:2]) # [N,M,2]
    rb = np.minimum(boxes1.reshape([num_box1, 1, -1])[:,:,2:], boxes2.reshape([1, num_box2, -1])[:,:,2:]) # [N,M,2]

    wh = (rb - lt + to_move).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    return inter

def correct_img_info(img_dir, image_file):
    with open(image_file, 'r') as f:
        data = json.load(f)
    for i in range(len(data)):
        img = data[i]
        basename = '{}.jpg'.format(img['image_id'])
        filename = os.path.join(img_dir, basename)
        img_data = Image.open(filename).convert("RGB")
        if img['width'] != img_data.size[0] or img['height'] != img_data.size[1]:
            print('--------- False id: ', i, '---------')
            print(img_data.size)
            print(img)
            data[i]['width'] = img_data.size[0]
            data[i]['height'] = img_data.size[1]
    with open(image_file, 'w') as outfile:  
        json.dump(data, outfile)

def load_info(dict_file, add_bg=True, predicate_set: str = "vg"):
    """
    Loads the file containing the visual genome label meanings
    """
    info = json.load(open(dict_file, 'r'))
    if add_bg:
        info['label_to_idx']['__background__'] = 0
        info['predicate_to_idx']['__background__'] = 0
        info['attribute_to_idx']['__background__'] = 0

    class_to_ind = info['label_to_idx']
    if (predicate_set or "").strip().lower() in {"vg", "visual_genome", "visual-genome"}:
        predicate_to_ind = info['predicate_to_idx']
    else:
        # Fashion-context predicate ontology for fashion scene style analysis.
        # NOTE: when training, your roidb_file H5 must be encoded with the same predicate indices.
        from maskrcnn_benchmark.data.datasets.fashion_predicates import get_predicate_set
        ind_to_predicates_override = get_predicate_set(predicate_set)
        predicate_to_ind = {p: i for i, p in enumerate(ind_to_predicates_override)}
    attribute_to_ind = info['attribute_to_idx']
    ind_to_classes = sorted(class_to_ind, key=lambda k: class_to_ind[k])
    ind_to_predicates = sorted(predicate_to_ind, key=lambda k: predicate_to_ind[k])
    ind_to_attributes = sorted(attribute_to_ind, key=lambda k: attribute_to_ind[k])

    return ind_to_classes, ind_to_predicates, ind_to_attributes


def load_image_filenames(img_dir, image_file):
    """
    Loads the image filenames from visual genome from the JSON file that contains them.
    This matches the preprocessing in scene-graph-TF-release/data_tools/vg_to_imdb.py.
    Parameters:
        image_file: JSON file. Elements contain the param "image_id".
        img_dir: directory where the VisualGenome images are located
    Return: 
        List of filenames corresponding to the good images
    """
    with open(image_file, 'r') as f:
        im_data = json.load(f)

    corrupted_ims = ['1592.jpg', '1722.jpg', '4616.jpg', '4617.jpg']
    fns = []
    img_info = []
    for i, img in enumerate(im_data):
        basename = '{}.jpg'.format(img['image_id'])
        if basename in corrupted_ims:
            continue

        filename = os.path.join(img_dir, basename)
        if os.path.exists(filename):
            fns.append(filename)
            img_info.append(img)
    assert len(fns) == 108073
    assert len(img_info) == 108073
    return fns, img_info


def load_graphs(
    roidb_file,
    split,
    num_im,
    num_val_im,
    filter_empty_rels,
    filter_non_overlap,
    skip_h5_train_val_carve=False,
    strict_stratified_ids=None,
    id_per_h5_row=None,
):
    """
    Load the file containing the GT boxes and relations, as well as the dataset split
    Parameters:
        roidb_file: HDF5
        split: (train, val, or test)
        num_im: Number of images we want
        num_val_im: Number of validation images
        filter_empty_rels: (will be filtered otherwise.)
        filter_non_overlap: If training, filter images that dont overlap.
        strict_stratified_ids: If set, select rows only by these image_id strings (ignore HDF5 split flags).
        id_per_h5_row: image_id per HDF5 row (same length as split), from image_data.json order used in load_image_filenames.
    Return: 
        image_index: numpy array corresponding to the index of images we're using
        boxes: List where each element is a [num_gt, 4] array of ground 
                    truth boxes (x1, y1, x2, y2)
        gt_classes: List where each element is a [num_gt] array of classes
        relationships: List where each element is a [num_r, 3] array of 
                    (box_ind_1, box_ind_2, predicate) relationships
    """
    roi_h5 = h5py.File(roidb_file, 'r')
    data_split = roi_h5['split'][:]
    if strict_stratified_ids is not None:
        if id_per_h5_row is None:
            raise ValueError("load_graphs: id_per_h5_row is required when strict_stratified_ids is set")
        n = int(data_split.shape[0])
        if len(id_per_h5_row) != n:
            raise RuntimeError(
                "STYLE strict split: len(image ids from image_file)=%s != len(h5 split)=%s. "
                "image_data.json / roidb alignment is broken for this dataset."
                % (len(id_per_h5_row), n)
            )
        allowed = strict_stratified_ids
        split_mask = np.zeros(n, dtype=bool)
        img_first_box = roi_h5['img_to_first_box'][:]
        img_first_rel = roi_h5['img_to_first_rel'][:]
        for i in range(n):
            if id_per_h5_row[i] not in allowed:
                continue
            if img_first_box[i] < 0:
                continue
            if filter_empty_rels and img_first_rel[i] < 0:
                continue
            split_mask[i] = True
        image_index = np.where(split_mask)[0]
        if num_im > -1:
            image_index = image_index[:num_im]
        split_mask = np.zeros_like(data_split).astype(bool)
        split_mask[image_index] = True
    else:
        split_flag = 2 if split == 'test' else 0
        split_mask = data_split == split_flag

        # Filter out images without bounding boxes
        split_mask &= roi_h5['img_to_first_box'][:] >= 0
        if filter_empty_rels:
            split_mask &= roi_h5['img_to_first_rel'][:] >= 0

        image_index = np.where(split_mask)[0]
        if num_im > -1:
            image_index = image_index[:num_im]
        if num_val_im > 0 and not skip_h5_train_val_carve:
            if split == 'val':
                image_index = image_index[:num_val_im]
            elif split == 'train':
                image_index = image_index[num_val_im:]

        split_mask = np.zeros_like(data_split).astype(bool)
        split_mask[image_index] = True

    # Get box information
    all_labels = roi_h5['labels'][:, 0]
    all_attributes = roi_h5['attributes'][:, :]
    all_boxes = roi_h5['boxes_{}'.format(BOX_SCALE)][:]  # cx,cy,w,h
    assert np.all(all_boxes[:, :2] >= 0)  # sanity check
    assert np.all(all_boxes[:, 2:] > 0)  # no empty box

    # convert from xc, yc, w, h to x1, y1, x2, y2
    all_boxes[:, :2] = all_boxes[:, :2] - all_boxes[:, 2:] / 2
    all_boxes[:, 2:] = all_boxes[:, :2] + all_boxes[:, 2:]

    im_to_first_box = roi_h5['img_to_first_box'][split_mask]
    im_to_last_box = roi_h5['img_to_last_box'][split_mask]
    im_to_first_rel = roi_h5['img_to_first_rel'][split_mask]
    im_to_last_rel = roi_h5['img_to_last_rel'][split_mask]

    # load relation labels
    _relations = roi_h5['relationships'][:]
    _relation_predicates = roi_h5['predicates'][:, 0]
    assert (im_to_first_rel.shape[0] == im_to_last_rel.shape[0])
    assert (_relations.shape[0] == _relation_predicates.shape[0])  # sanity check

    # Get everything by image.
    boxes = []
    gt_classes = []
    gt_attributes = []
    relationships = []
    for i in range(len(image_index)):
        i_obj_start = im_to_first_box[i]
        i_obj_end = im_to_last_box[i]
        i_rel_start = im_to_first_rel[i]
        i_rel_end = im_to_last_rel[i]

        boxes_i = all_boxes[i_obj_start : i_obj_end + 1, :]
        gt_classes_i = all_labels[i_obj_start : i_obj_end + 1]
        gt_attributes_i = all_attributes[i_obj_start : i_obj_end + 1, :]

        if i_rel_start >= 0:
            predicates = _relation_predicates[i_rel_start : i_rel_end + 1]
            obj_idx = _relations[i_rel_start : i_rel_end + 1] - i_obj_start # range is [0, num_box)
            assert np.all(obj_idx >= 0)
            assert np.all(obj_idx < boxes_i.shape[0])
            rels = np.column_stack((obj_idx, predicates)) # (num_rel, 3), representing sub, obj, and pred
        else:
            assert not filter_empty_rels
            rels = np.zeros((0, 3), dtype=np.int32)

        if filter_non_overlap:
            assert split == 'train'
            # construct BoxList object to apply boxlist_iou method
            # give a useless (height=0, width=0)
            boxes_i_obj = BoxList(boxes_i, (1000, 1000), 'xyxy')
            inters = boxlist_iou(boxes_i_obj, boxes_i_obj)
            rel_overs = inters[rels[:, 0], rels[:, 1]]
            inc = np.where(rel_overs > 0.0)[0]

            if inc.size > 0:
                rels = rels[inc]
            else:
                split_mask[image_index[i]] = 0
                continue

        boxes.append(boxes_i)
        gt_classes.append(gt_classes_i)
        gt_attributes.append(gt_attributes_i)
        relationships.append(rels)

    return split_mask, boxes, gt_classes, gt_attributes, relationships
