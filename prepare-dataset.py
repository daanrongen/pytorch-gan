def main(args):
    images = glob.glob(f"{args.dataset_path}/*.*")
    os.makedirs(args.output_path, exist_ok=True)

    for i, infile in tqdm(enumerate(images)):
        image = (
            Image.open(infile)
            .convert("RGB")
            .resize((args.size, args.size), Image.LANCZOS)
        )
        image = remove_frames(image)
        image = center_crop(image, args.size)
        assert image.size == (args.size, args.size)
        assert image.mode == "RGB"
        image.save(f"{args.output_path}/{i:04d}.png", "PNG", quality=100)


if __name__ == "__main__":
    import glob
    from utils import remove_frames, center_crop
    from PIL import Image
    import argparse
    import os
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description="GAN")
    parser.add_argument("--size", type=int, default=1024)
    args = parser.parse_args()
    args.dataset_path = "/content/drive/MyDrive/data/gan/images/teletext"
    args.output_path = f"content/drive/MyDrive/data/gan/datasets/{args.size}"

    main(args)
