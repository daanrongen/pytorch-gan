import glob
from PIL import Image
import argparse
import os
from tqdm import tqdm

from utils import remove_frames, center_crop, flip

def main(args):
    images = glob.glob(f"{args.indir}/*.*")
    os.makedirs(args.outdir, exist_ok=True)

    for i, infile in tqdm(enumerate(images)):
        image = (
            Image.open(infile)
            .convert("RGB")
            .resize((args.size, args.size), Image.LANCZOS)
        )
        image = remove_frames(image)
        image = center_crop(image, args.size)
        flipped = flip(image)
        
        assert image.size == (args.size, args.size)
        assert image.mode == "RGB"
        assert flipped.size == (args.size, args.size)
        assert flipped.mode == "RGB"

        image.save(f"{args.outdir}/{i:04d}.png", "PNG", quality=100)
        flipped.save(f"{args.outdir}/{i:04d}-flip.png", "PNG", quality=100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAN")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--indir", type=str, default="/content/drive/MyDrive/data/gan/images/in")
    parser.add_argument("--outdir", type=str, default="/content/drive/MyDrive/data/gan/images/out")
    args = parser.parse_args()

    main(args)
