import os, cv2, random, math
import numpy as np
from random import randrange
from intervaltree import IntervalTree


class StickerDataset:
  def __init__(self, folder_path, sticker_avg_size):
    self.imgs = []
    self.sequence = []
    self.sticker_avg_size = sticker_avg_size
    subdirs = next(os.walk(folder_path))[1]
    for subdir in subdirs:
      img_files = next(os.walk(folder_path + '/' + subdir))[2]
      for img_file in img_files:
        img = cv2.imread(folder_path + '/' + subdir + '/' + img_file, -1)
        self.imgs.append((img, int(subdir)))

  def get_item(self):
    if len(self.sequence) == 0:
      self.sequence = list(range(len(self.imgs)))
      random.shuffle(self.sequence)
    selection = self.imgs[self.sequence[-1]]
    return (sticker_augmentation(selection[0],self.sticker_avg_size),
            selection[1])

  def pop_item(self):
    self.sequence.pop()


class V_edge: # vertical edge
  def __init__(self, x, ymin, ymax, is_left):
    self.x = x
    self.ymin = ymin
    self.ymax = ymax
    self.is_left = is_left

  @classmethod
  def from_box(cls, box): # box is a numpy array, [cls, xmin, ymin, xmax, ymax]
    return (V_edge(box[1], box[2], box[4], True), V_edge(box[3], box[2], box[4], False))


def find_space(labels, img_width, img_height, sticker_width, sticker_height):
  # process edge
  ves = [] # vertical edges
  for label in labels:
    ve1, ve2 = V_edge.from_box(label)
    ves.append(ve1)
    ves.append(ve2)
  ves.append(V_edge(img_width, 0, img_height, True)) # right border
  ves.sort(key=lambda k: k.x)
  # start finding
  it = IntervalTree()
  it[0:img_height] = 0
  for ve in ves:
    # find legal box
    rects = []
    for top_y, bottom_y, leftmost_x in sorted(it):
      if leftmost_x < 0:
        rects.append((top_y, bottom_y, 0))
      else:
        rects.append((top_y, bottom_y, ve.x - leftmost_x))
    rect_cnt = len(rects)
    top = [None] * rect_cnt
    top[0] = 0
    # calculate top
    for i in range(1, rect_cnt):
      j = i - 1
      while j >= 0:
        if rects[j][2] < rects[i][2]:
          break
        j = top[j] - 1
      top[i] = j + 1
    # calculate bottom
    bottom = [None] * rect_cnt
    bottom[rect_cnt-1] = rect_cnt-1
    for i in range(rect_cnt-2, -1, -1):
      j = i + 1
      while j < rect_cnt:
        if rects[j][2] < rects[i][2]:
          break
        j = bottom[j] + 1
      bottom[i] = j - 1
    # calculate area
    for i in range(0, rect_cnt):
      curr_rect_w = rects[i][2]
      curr_rect_h = rects[bottom[i]][1] - rects[top[i]][0]
      if curr_rect_w >= sticker_width and curr_rect_h >= sticker_height:
        x_offset = int(curr_rect_w - sticker_width)
        y_offset = int(curr_rect_h - sticker_height)
        return (ve.x - curr_rect_w + randrange(x_offset+1), rects[top[i]][0] + randrange(y_offset+1))
    # update interval
    for r in sorted(it[ve.ymin:ve.ymax]):
      original_data = r.data
      # find new_begin, new_end
      new_begin = max(ve.ymin, r.begin)
      new_end = min(ve.ymax, r.end)
      it.chop(new_begin, new_end)
      if ve.is_left:
        new_data = min(-1, original_data-1)
      else:
        if original_data < -1:
          new_data = original_data + 1
        else:
          new_data = ve.x
      it.addi(new_begin, new_end, new_data)
    # merge same inteval
    it_list = sorted(it)
    prev_begin = it_list[0].begin
    prev_data = it_list[0].data
    merge_cnt = 1
    for r in it_list[1:]:
      if r.data != prev_data:
        if merge_cnt > 1:
          it.remove_envelop(prev_begin, r.begin)
          it.addi(prev_begin, r.begin, prev_data)
        prev_begin = r.begin
        prev_data = r.data
        merge_cnt = 1
      else:
        merge_cnt += 1
    if merge_cnt > 1:
      it.remove_envelop(prev_begin, img_height)
      it.addi(prev_begin, img_height, prev_data)
  return None


def overlay_image(target_img, source_img, x_offset, y_offset, labels, source_class):
  source_x1 = source_y1 = 0
  source_x2 = source_img.shape[1]
  source_y2 = source_img.shape[0]
  x1, x2 = x_offset, x_offset + source_img.shape[1]
  y1, y2 = y_offset, y_offset + source_img.shape[0]
  if x1 < 0:
    source_x1 = -x1
    x1 = 0
  if y1 < 0:
    source_y1 = -y1
    y1 = 0
  if x2 > target_img.shape[1]:
    source_x2 -= x2 - target_img.shape[1]
    x2 = target_img.shape[1]
  if y2 > target_img.shape[0]:
    source_y2 -= y2 - target_img.shape[0]
    y2 = target_img.shape[0]
  alpha_s = source_img[source_y1:source_y2, source_x1:source_x2, 3] / 255.0
  alpha_l = 1.0 - alpha_s
  for c in range(0, 3):
    target_img[y1:y2, x1:x2, c] = (alpha_s * source_img[source_y1:source_y2, source_x1:source_x2, c] +
                                   alpha_l * target_img[y1:y2, x1:x2, c])
  return np.vstack((labels, np.array([source_class, x1, y1, x2, y2])))


def rotate_image(mat, angle):
  height, width = mat.shape[:2] # image shape has 3 dimensions
  image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

  rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

  # rotation calculates the cos and sin, taking absolutes of those.
  abs_cos = abs(rotation_mat[0,0]) 
  abs_sin = abs(rotation_mat[0,1])

  # find the new width and height bounds
  bound_w = int(height * abs_sin + width * abs_cos)
  bound_h = int(height * abs_cos + width * abs_sin)

  # subtract old image center (bringing image back to origo) and adding the new image center coordinates
  rotation_mat[0, 2] += bound_w/2 - image_center[0]
  rotation_mat[1, 2] += bound_h/2 - image_center[1]

  # rotate image with the new bounds and translated rotation matrix
  rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
  return rotated_mat


def sticker_augmentation(img, sticker_avg_size, degrees=10, scale=0.2, shear=5):
  h0, w0 = img.shape[0], img.shape[1]
  # Flip
  if random.random() < 0.5:
    img = cv2.flip(img, 1)
  # Shear
  S = np.eye(3)[:2]
  S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
  S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)
  S[0, 2] = -S[0, 1] * w0/2
  S[1, 2] = -S[1, 0] * h0/2
  img = cv2.warpAffine(img, S, dsize=(w0, h0))
  # Scale
  random_scale_factor = random.uniform(1 - scale, 1 + scale)
  scale_factor = sticker_avg_size / max(h0, w0)
  if scale_factor < 1: # resize down
    scale_factor *= random_scale_factor
  else:
    scale_factor = random_scale_factor
  img = cv2.resize(img, dsize=(0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
  # Rotate
  img = rotate_image(img, random.uniform(-degrees, degrees))
  # Crop Transparent
  img = crop_transparent(img)
  return img
 
 
def crop_transparent(img):
  sticker_h, sticker_w = img.shape[:2]
  left_b = top_b = 0
  right_b = sticker_w
  bottom_b = sticker_h
  for i in range(sticker_w):
    left_b = i
    if not (img[:, i, 3] == 0).all():
      break
  for i in range(sticker_w - 1, -1, -1):
    if not (img[:, i, 3] == 0).all():
      break
    right_b = i
  for i in range(sticker_h):
    top_b = i
    if not (img[i, :, 3] == 0).all():
      break
  for i in range(sticker_h - 1, -1, -1):
    if not (img[i, :, 3] == 0).all():
      break
    bottom = i
  return img[top_b:bottom_b, left_b:right_b, :]