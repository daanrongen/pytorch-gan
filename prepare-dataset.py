import glob
from PIL import Image
import argparse
import os
from tqdm import tqdm

from utils import remove_frames, center_crop, flip

# def crop_center(pil_img, crop_width, crop_height):
#     img_width, img_height = pil_img.size
#     return pil_img.crop(((img_width - crop_width) // 2,
#                          (img_height - crop_height) // 2,
#                          (img_width + crop_width) // 2,
#                          (img_height + crop_height) // 2))

# def crop_max_square(pil_img):
#     return crop_center(pil_img, min(pil_img.size), min(pil_img.size))

def main(args):
    images = glob.glob(f"{args.indir}/*.*")
    os.makedirs(args.outdir, exist_ok=True)

    for i, infile in tqdm(enumerate(images)):
        image = (
            Image.open(infile)
            .convert("RGB")
            .resize((args.size, args.size), Image.LANCZOS)
        )

        # image = crop_max_square(image).resize((args.size, args.size), Image.LANCZOS)
        image = remove_frames(image)
        image = center_crop(image, args.size)

        assert image.size == (args.size, args.size)
        assert image.mode == "RGB"
        image.save(f"{args.outdir}/{i:06d}.png", "PNG", quality=100)

        if args.flip:
            flipped = flip(image)
            assert flipped.size == (args.size, args.size)
            assert flipped.mode == "RGB"
            flipped.save(f"{args.outdir}/{i:06d}-flip.png", "PNG", quality=100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAN")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--indir", type=str, default="/content/drive/MyDrive/data/gan/images/in")
    parser.add_argument("--outdir", type=str, default="/content/drive/MyDrive/data/gan/images/out")
    parser.add_argument("--flip", type=bool, default=False)
    args = parser.parse_args()

    main(args)
