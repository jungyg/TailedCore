import os
from tqdm import tqdm
import csv

SOURCE_ROOT = "./data" # source root of visa
TARGET_ROOT = "./data" # target root of visa

def restructure_visa():

    source_dir = os.path.join(SOURCE_ROOT, "visa")
    target_dir = os.path.join(TARGET_ROOT, "visa_")

    os.makedirs(target_dir, exist_ok = True)

    with open(os.path.join(source_dir, "split_csv", "1cls.csv")) as file:
        csvreader = csv.reader(file)
        header = next(csvreader)

        for row_no, row_content in tqdm(enumerate(csvreader)):
            
            class_, split_, label_, img_src, mask_src = row_content
            img_no = img_src.split('/')[-1]
            mask_no = mask_src.split('/')[-1] if mask_source != "" else ""
            label_ = "good" if label_=="normal" else "anomaly"

            if not os.path.exists(os.path.join(target_dir, class_, split_, label_)):
                os.makedirs(os.path.join(target_dir, class_, split_, label_), exist_ok=True)

            image_source = os.path.join(source_dir, image_source)
            image_target = os.path.join(target_dir, class_, split_, label_, img_no)
            assert ".JPG" in image_target, "wrong image target"
            image_target = image_target.replace(".JPG", ".png")
            os.symlink(image_source, image_target)

            if mask_source == "":   # good class
                continue

            else:   # anomal class
                assert split_=="test" and label_=="anomaly"

                if not os.path.exists(os.path.join(target_dir, class_, "ground_truth", label_)):
                    os.makedirs(os.path.join(target_dir, class_, "ground_truth", label_), exist_ok=True)
                mask_source = os.path.join(source_dir, mask_source)
                mask_target = os.path.join(target_dir, class_, "ground_truth", label_, mask_no)
                assert ".png" in mask_target, "wrong mask target"
                mask_target = mask_target.replace(".png", "_mask.png")
                os.symlink(mask_source, mask_target)





if __name__ == "__main__":
    restructure_visa()