import os
import re
import sys

data_path = r"C:\Users\Administrator\Desktop\Capstone Workshop-Edge Computing Device Programming for AI Projects\Asssignment\Assignment 2\Data"

labels = os.path.join(data_path, 'labels.txt')

Train_dir = os.path.join(data_path, 'train')
Test_dir = os.path.join(data_path, 'test')
Val_dir = os.path.join(data_path, 'val')

checkTrain = os.path.isdir(Train_dir)
checkTest = os.path.isdir(Test_dir)
checkVal = os.path.isdir(Val_dir)

CLASSES = ['elephant', 'monkey', 'rabbit']
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
EXPECTED_SPLITS = {'train': 21, 'val': 6, 'test': 3}
EXPECTED_TOTAL_PER_CLASS = sum(EXPECTED_SPLITS.values())  # 30

if not os.path.isfile(labels):
    os.makedirs(data_path, exist_ok=True)
    with open(labels, 'w', encoding='utf-8') as f:
        f.write('\n'.join(CLASSES) + '\n')
    print(f"Created labels.txt at: {labels}")
else:
    print(labels, 'does exist')

def is_image_file(path: str) -> bool:
    return os.path.isfile(path) and os.path.splitext(path)[1].lower() in IMG_EXTS

def list_images(dir_path: str):
    try:
        return [f for f in os.listdir(dir_path) if is_image_file(os.path.join(dir_path, f))]
    except FileNotFoundError:
        return []


if checkTrain:
    print(Train_dir, " does exist")
    for cls in CLASSES:
        found = os.path.isdir(os.path.join(Train_dir, cls))
        if found:
            print(os.path.join(Train_dir, cls), " does exist")
        else:
            print(f"Missing train/{cls} directory!")
else:
    print("Missing train directory!")

if checkTest:
    print(Test_dir, " does exist")
    for cls in CLASSES:
        found = os.path.isdir(os.path.join(Test_dir, cls))
        if found:
            print(os.path.join(Test_dir, cls), " does exist")
        else:
            print(f"Missing test/{cls} directory!")
else:
    print("Missing test directory!")

if checkVal:
    print(Val_dir, " does exist")
    for cls in CLASSES:
        found = os.path.isdir(os.path.join(Val_dir, cls))
        if found:
            print(os.path.join(Val_dir, cls), " does exist")
        else:
            print(f"Missing val/{cls} directory!")
else:
    print("Missing val directory!")


ok = True
class_totals = {c: 0 for c in CLASSES}

def check_filename_pattern(files, label: str, split: str, dir_path: str):
    pat = re.compile(rf"^{re.escape(label)}\s+{re.escape(split)}\s*\((\d+)\)$", re.IGNORECASE)
    warnings = 0
    numbers = set()
    for fname in files:
        stem, _ = os.path.splitext(fname)
        m = pat.match(stem)
        if not m:
            print(f"WARNING: Unexpected filename pattern in {dir_path}: '{fname}' "
                  f"(expected like: '{label} {split} (1).jpg').")
            warnings += 1
        else:
            try:
                numbers.add(int(m.group(1)))
            except ValueError:
                print(f"WARNING: Could not parse number in: '{fname}'")
                warnings += 1

    expected_n = EXPECTED_SPLITS[split]
    if len(numbers) == expected_n:
        missing = sorted(set(range(1, expected_n + 1)) - numbers)
        if missing:
            print(f"WARNING: '{label} {split}' missing numbers: {missing}")
        if 1 not in numbers:
            print(f"WARNING: '{label} {split}' numbers do not start at 1.")
    return warnings

for cls in CLASSES:
    for split, expected_count in EXPECTED_SPLITS.items():
        dir_path = os.path.join(data_path, split, cls)
        if not os.path.isdir(dir_path):
            ok = False
            continue

        files = list_images(dir_path)
        n = len(files)
        class_totals[cls] += n

        if n != expected_count:
            print(f"ERROR: {dir_path} has {n} images, expected {expected_count}.")
            ok = False
        else:
            print(f"OK: {dir_path} has {n} images (expected {expected_count}).")

        check_filename_pattern(files, cls, split, dir_path)


for cls in CLASSES:
    total = class_totals[cls]
    if total != EXPECTED_TOTAL_PER_CLASS:
        print(f"ERROR: Total images for '{cls}' = {total}, expected {EXPECTED_TOTAL_PER_CLASS}.")
        ok = False
    else:
        print(f"OK: Total images for '{cls}' = {total}.")


if not ok:
    sys.exit(1)