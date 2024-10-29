import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils

def plot_images(batch, device, n):
    """Plot a grid of real images."""
    img_example = batch.float().to(device)[:n]
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.title('Training Images')
    plt.imshow(np.transpose(vutils.make_grid(img_example, padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.savefig('output/image_example.png')

def plot_loss_history(G_losses, D_losses):
    """Plot generator and discriminator loss values."""
    plt.figure(figsize=(10, 5))
    plt.title('Generator and Discriminator Loss Values')
    plt.plot(G_losses, label='G')
    plt.plot(D_losses, label='D')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('output/loss_history.png')

def plot_image_comparison(img, batch, device, n):
    """Plot a comparison of real and fake images."""
    img_example = vutils.make_grid(batch.to(device)[:n], padding=5, normalize=True)
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis('off')
    plt.title('Real Images')
    plt.imshow(np.transpose(img_example.cpu(), (1, 2, 0)))

    plt.subplot(1, 2, 2)
    plt.axis('off')
    plt.title('Fake Images')
    plt.imshow(img)
    plt.savefig('output/image_comparison.png')

def save_output(epoch, netG, fixed_noise):
    """Save the output of a generator."""
    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()

    img = (np.transpose(vutils.make_grid(fake, padding=2, normalize=True).numpy(), (1, 2, 0)) * 255.0).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)

    cv2.imwrite(f'output/images/img_{epoch + 1}.png', img)
    return img

def get_opencv_image(img):
    """Get an opencv image."""
    low, high = img.min(), img.max()

    img.sub_(low).div_(max(high - low, 1e-5))
    img = (np.transpose(img.numpy(), (1, 2, 0)) * 255.0).astype(np.uint8)

    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
    return img
