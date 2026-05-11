import os
import random

IMG_DIR = "dataset_landcover/output/images"
MASK_DIR = "dataset_landcover/output/masks"

out_train = "dataset_landcover/output/train.txt"
out_val   = "dataset_landcover/output/val.txt"
out_test  = "dataset_landcover/output/test.txt"

all_imgs = [f for f in os.listdir(IMG_DIR) if f.endswith(".png")]

valid = []
for img in all_imgs:
    mask = os.path.join(MASK_DIR, img)
    if os.path.exists(mask):
        valid.append(img)
    else:
        print("no mask for :", img)


# shuffle
random.shuffle(valid)

n = len(valid)
train = valid[:int(0.7*n)]
val   = valid[int(0.7*n):int(0.9*n)]
test  = valid[int(0.9*n):]

with open(out_train, "w") as f:
    for x in train:
        f.write(x + "\n")

with open(out_val, "w") as f:
    for x in val:
        f.write(x + "\n")

with open(out_test, "w") as f:
    for x in test:
        f.write(x + "\n")

print("[INFO] DONE!")
print(f" Train = {len(train)}")
print(f" Val   = {len(val)}")
print(f" Test  = {len(test)}")
