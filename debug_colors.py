import cv2, numpy as np, os, glob
from PIL import Image

ref_dir = r'C:\Users\mido\Documents\antigravity\focused-babbage\reference\country'
ref_inst = r'C:\Users\mido\Documents\antigravity\focused-babbage\reference\institution'
img15 = cv2.imread(r'C:\Users\mido\Documents\antigravity\focused-babbage\validate_ai\15.jpg')

K = 5

def dominant_lab(img, k=K):
    small = cv2.resize(img, (60, 40), interpolation=cv2.INTER_AREA)
    pix = small.reshape(-1, 3).astype(np.float32)
    if len(pix) < k:
        pix = np.tile(small.mean(axis=(0,1)).reshape(1, 3), (k, 1))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    _, labels, centers = cv2.kmeans(pix, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=k)
    order = np.argsort(-counts)
    cb = centers[order].astype(np.uint8).reshape(1, k, 3)
    return cv2.cvtColor(cb, cv2.COLOR_BGR2Lab).reshape(k, 3).astype(np.float32)

def pdist(c1, c2):
    k = len(c1)
    dm = np.linalg.norm(c1[:, None, :] - c2[None, :, :], axis=2)
    used = set()
    total = 0.0
    for i in range(k):
        bj = min((j for j in range(k) if j not in used), key=lambda j: dm[i, j])
        total += dm[i, bj]
        used.add(bj)
    return total / k

def isolate(crop, wv=200, ws=60, it=5):
    h, w = crop.shape[:2]
    if h < 10 or w < 10:
        return crop
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    white = (hsv[:, :, 2] > wv) & (hsv[:, :, 1] < ws)
    if not white.any():
        mh, mw = max(1, h // 5), max(1, w // 5)
        return crop[mh:h-mh, mw:w-mw]
    rows = white.any(axis=1)
    cols = white.any(axis=0)
    wy1 = int(np.where(rows)[0][0])
    wy2 = int(np.where(rows)[0][-1])
    wx1 = int(np.where(cols)[0][0])
    wx2 = int(np.where(cols)[0][-1])
    yi1, yi2 = wy1 + it, wy2 - it + 1
    xi1, xi2 = wx1 + it, wx2 - it + 1
    if yi2 <= yi1 + 3 or xi2 <= xi1 + 3:
        return crop
    r = crop[yi1:yi2, xi1:xi2]
    return r if r.size >= 30 else crop

print('Loading templates...')
tmpl = {}
for p in glob.glob(os.path.join(ref_dir, '*.png')):
    n = os.path.splitext(os.path.basename(p))[0]
    img = cv2.imread(p)
    if img is not None:
        tmpl[n] = dominant_lab(img)

for p in glob.glob(os.path.join(ref_inst, '*')):
    try:
        pil = Image.open(p).convert('RGB')
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        n = os.path.splitext(os.path.basename(p))[0]
        tmpl[n] = dominant_lab(img)
    except Exception:
        pass

print(f'Loaded {len(tmpl)} templates')

gt = {
    'de': (1065, 124, 1103, 162),
    'ru': (1173, 2929, 1211, 2969),
    'fr': (2110, 1361, 2172, 1427),
}

for code, (x1, y1, x2, y2) in gt.items():
    crop = img15[y1:y2, x1:x2]
    interior = isolate(crop)
    cc = dominant_lab(interior)
    dists = {n: pdist(cc, tc) for n, tc in tmpl.items()}
    ranked = sorted(dists.items(), key=lambda x: x[1])
    rank = next(i for i, (n, _) in enumerate(ranked) if n == code) + 1
    top5 = [(n, round(d, 1)) for n, d in ranked[:5]]
    de_d = round(dists.get('de', 999), 1)
    ru_d = round(dists.get('ru', 999), 1)
    fr_d = round(dists.get('fr', 999), 1)
    print(f'crop_{code}: rank={rank}/319, top5={top5}')
    print(f'  de={de_d}, ru={ru_d}, fr={fr_d}')
