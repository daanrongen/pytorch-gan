def main(args):
    dataset = datasets.ImageFolder(root=args.dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    gen = Generator(args.noise_dim, args.channels_img, args.features_gen).to(device)
    disc = Discriminator(args.channels_img, args.features_disc).to(device)
    initialize_weights(gen)
    initialize_weights(disc)

    opt_gen = optim.Adam(gen.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    fixed_noise = torch.randn(32, args.noise_dim, 1, 1).to(device)
    writer_real = SummaryWriter("logs/real")
    writer_fake = SummaryWriter("logs/fake")
    step = 0

    gen.train()
    disc.train()

    for epoch in range(args.epochs):
        for batch_idx, (real, _) in enumerate(dataloader):
            real = real.to(device)
            noise = torch.randn(args.batch_size, args.noise_dim, 1, 1).to(device)
            fake = gen(noise)

            ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
            disc_real = disc(real).reshape(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake.detach()).reshape(-1)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
            output = disc(fake).reshape(-1)
            loss_gen = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            # Print losses occasionally and print to tensorboard
            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch}/{args.epochs}] Batch {batch_idx}/{len(dataloader)} \
                          Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
                )

                with torch.no_grad():
                    fake = gen(fixed_noise)
                    img_grid_real = torchvision.utils.make_grid(
                        real[:32], normalize=True
                    )
                    img_grid_fake = torchvision.utils.make_grid(
                        fake[:32], normalize=True
                    )

                    writer_real.add_image("Real", img_grid_real, global_step=step)
                    writer_fake.add_image("Fake", img_grid_fake, global_step=step)

                step += 1


if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from model import Discriminator, Generator, initialize_weights
    import argparse

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description="GAN")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--noise-dim", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    args.dataset_path = f"/content/drive/MyDrive/data/gan/datasets/{args.image_size}/"
    args.channels_img = 3
    args.features_disc = 64
    args.features_gen = 64
    args.learning_rate = 2e-4

    transforms = transforms.Compose(
        [
            transforms.Resize(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5 for _ in range(args.channels_img)],
                [0.5 for _ in range(args.channels_img)],
            ),
        ]
    )

    main(args)
