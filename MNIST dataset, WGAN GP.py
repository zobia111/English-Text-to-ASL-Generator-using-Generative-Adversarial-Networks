#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.models import inception_v3
from scipy.stats import entropy

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt


# In[ ]:


import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image

# Load sign language MNIST dataset
train = pd.read_csv("sign_mnist_train.csv")
test = pd.read_csv("sign_mnist_test.csv")

Y_train = train["label"]
Y_test = test["label"]
X_train = train.drop(labels=["label"], axis=1)
X_test = test.drop(labels=["label"], axis=1)

# Custom dataset class for sign language MNIST
class SignLanguageDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = np.array(self.data.iloc[idx, 0])
        image = np.array(self.data.iloc[idx, 1:]).reshape(28, 28, 1).astype(np.uint8)

        if self.transform:
            image = self.transform(image)

        return image, label

# Load sign language MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
sign_language_train_dataset = SignLanguageDataset(data=train, transform=transform)
sign_language_test_dataset = SignLanguageDataset(data=test, transform=transform)

# Create data loaders for sign language MNIST
batch_size = 64
sign_language_train_dataloader = DataLoader(dataset=sign_language_train_dataset, batch_size=batch_size, shuffle=True)
sign_language_test_dataloader = DataLoader(dataset=sign_language_test_dataset, batch_size=batch_size, shuffle=False)



# Hyperparameters
n_epochs = 1000
batch_size = 64
lr = 0.0002
b1 = 0.5
b2 = 0.999
n_cpu = 8
latent_dim = 100
img_size = 28
channels = 1
n_critic = 5
clip_value = 0.01
sample_interval = 400

img_shape = (channels, img_size, img_size)

cuda = True if torch.cuda.is_available() else False

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity
# Loss weight for gradient penalty
lambda_gp = 10

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()


# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


os.makedirs("generated_images", exist_ok=True)

# Training loop
batches_done = 0
for epoch in range(n_epochs):
    for i, (imgs, _) in enumerate(sign_language_train_dataloader):

        real_imgs = Variable(imgs.type(Tensor))

        optimizer_D.zero_grad()

        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim))))

        fake_imgs = generator(z)

        # Switch discriminator to training mode
        discriminator.train()

        real_validity = discriminator(real_imgs)
        fake_validity = discriminator(fake_imgs)
        gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()

        if i % n_critic == 0:
            # Switch discriminator to evaluation mode
            discriminator.eval()

            fake_imgs = generator(z)
            fake_validity = discriminator(fake_imgs)
            g_loss = -torch.mean(fake_validity)

            g_loss.backward()
            optimizer_G.step()

            # Print losses
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, n_epochs, i, len(sign_language_train_dataloader), d_loss.item(), g_loss.item())
            )

            # Save generated images at specified intervals
            if batches_done % sample_interval == 0:
                save_image(fake_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

            batches_done += n_critic

    # Switch discriminator back to training mode
    discriminator.train()

    # Print and save generated images after each epoch
    print(f'\nGenerated Examples - Epoch {epoch + 1}:')
    n = 5
    plt.figure(figsize=(6, 4))
    for i in range(1, n + 1):
        ax = plt.subplot(1, n, i)
        noise = np.random.normal(0, 1, (1, 100))
        # Switch generator to evaluation mode
        generator.eval()
        generated_img = generator(Variable(torch.FloatTensor(noise).cuda() if cuda else torch.FloatTensor(noise))).cpu().detach().numpy()
        # Switch generator back to training mode
        generator.train()
        plt.imshow(generated_img.reshape(img_size, img_size), cmap='gray')
        plt.axis('off')
    plt.show()

    # Save generated images
    save_image(torch.FloatTensor(generated_img).view(1, channels, img_size, img_size),
               f"generated_images/epoch_{epoch + 1}.png", normalize=True)

