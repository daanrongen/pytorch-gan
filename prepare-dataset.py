import glob
import cv2
import argparse
import os
from skimage.util import random_noise
import numpy as np
from tqdm import tqdm


def main(args):
    images = glob.glob(f"{args.indir}/*.*")
    os.makedirs(args.outdir, exist_ok=True)

    for i, infile in tqdm(enumerate(images)):
        image = cv2.imread(infile)
        print(image.shape)

        image = cv2.resize(image, (args.size, args.size), cv2.INTER_CUBIC)
        print(image.shape)

        cv2.imwrite(f"{args.outdir}/{i:06d}-0.png", image)

        if (args.invert):
            cv2.imwrite(f"{args.outdir}/{i:06d}-clr.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        if (args.noise):
            noise = random_noise(image)
            noise = np.array(255 * noise, dtype=np.uint8)
            cv2.imwrite(f"{args.outdir}/{i:06d}-nse.png", noise)

        if (args.blur):
            blur_image = cv2.GaussianBlur(image, (11, 11), 0)
            cv2.imwrite(f"{args.outdir}/{i:06d}-blr.png", blur_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAN")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--indir", type=str, default="/content/drive/MyDrive/data/gan/images/in")
    parser.add_argument("--outdir", type=str, default="/content/drive/MyDrive/data/gan/images/out")
    parser.add_argument("--invert", type=bool, default=False)
    parser.add_argument("--blur", type=bool, default=False)
    parser.add_argument("--noise", type=bool, default=False)
    args = parser.parse_args()

    main(args)
