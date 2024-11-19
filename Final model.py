#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

from torchvision.utils import save_image
import matplotlib.pyplot as plt


# In[2]:


import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import numpy as np
import re
import torch

# Function to extract signer and class information from image file names
def extract_signer_and_class(image_file):
    # Use regular expression to extract signer and class information
    match = re.match(r"hand\d+_(\w+)_\w+_seg_\d+_cropped.png", image_file)
    if match:
        class_label = match.group(1)  # Extract class information
        return class_label
    else:
        return None

# Function to load and preprocess the dataset
def load_dataset(root_folder_path, target_size):
    data = []
    labels = []

    for signer_folder in os.listdir(root_folder_path):
        signer_folder_path = os.path.join(root_folder_path, signer_folder)

        for image_file in os.listdir(signer_folder_path):
            class_label = extract_signer_and_class(image_file)  # Extract class information

            if class_label is not None:
                image_path = os.path.join(signer_folder_path, image_file)
                img = Image.open(image_path)
                img = img.resize(target_size)

                data.append(np.array(img))
                labels.append(class_label)

    # Convert labels to numerical format
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # Reshape data to match the expected input shape of the discriminator
    data = np.transpose(data, (0, 3, 1, 2))  # Change data shape to (N, C, H, W)

    return torch.tensor(data, dtype=torch.float32), torch.tensor(labels_encoded)

# Load and preprocess the dataset
root_folder_path = 'ASL_Dataset_final'
target_size = (128, 128)
data, labels = load_dataset(root_folder_path, target_size)

# Print the size of the loaded dataset and unique classes
print("Size of the loaded dataset:", len(data))
unique_classes = torch.unique(labels)
print("Unique Classes:", unique_classes)

# Split the dataset into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

# Normalize pixel values to be between 0 and 1
train_data = train_data / 255.0
test_data = test_data / 255.0

# Display the unique classes in the training dataset
unique_train_classes = torch.unique(train_labels)
print("Unique Classes in Training Set:", unique_train_classes)


# In[3]:


# Function to load and preprocess the dataset with custom class mapping
def load_dataset_with_mapping(root_folder_path, target_size):
    data = []
    labels = []

    class_mapping = {}
    class_index = 0

    for signer_folder in os.listdir(root_folder_path):
        signer_folder_path = os.path.join(root_folder_path, signer_folder)

        for image_file in os.listdir(signer_folder_path):
            class_label = extract_signer_and_class(image_file)  # Extract class information

            if class_label is not None:
                if class_label not in class_mapping:
                    class_mapping[class_label] = class_index
                    class_index += 1

                image_path = os.path.join(signer_folder_path, image_file)
                img = Image.open(image_path)
                resized_img = img.resize(target_size)
                data.append(np.array(resized_img))
                labels.append(class_mapping[class_label])

    # Convert labels to numerical format
    labels = np.array(labels)

    return np.array(data), labels, class_mapping

# Load and preprocess the dataset with custom class mapping
root_folder_path = 'ASL_Dataset_final'
data, labels, class_mapping = load_dataset_with_mapping(root_folder_path, target_size)

# Print the size of the loaded dataset and unique classes
print("Size of the loaded dataset:", len(data))

# Print the unique custom classes mapping
print("Class Mapping:", class_mapping)


# In[4]:


# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Normalize pixel values to be between 0 and 1
X_train = X_train.astype('float32') / 255.0
X_test =  X_test.astype('float32') / 255.0

# Display the unique classes in the training dataset
unique_train_classes = np.unique(Y_train)
print("Unique Classes in Training Set:", unique_train_classes)


# In[5]:


# Create an inverse mapping from numerical labels to alphanumeric values
inverse_mapping = {v: k for k, v in class_mapping.items()}

# Convert numeric labels back to alphanumeric values
train_labels_decoded = np.vectorize(inverse_mapping.get)(train_labels)
test_labels_decoded = np.vectorize(inverse_mapping.get)(test_labels)

# Display the unique classes in the training dataset (decoded)
unique_train_classes_decoded = np.unique(train_labels_decoded)
print("Unique Classes in Training Set (Decoded):", unique_train_classes_decoded)


# In[6]:


import torch
from torch.utils.data import Dataset
from PIL import Image

class SignLanguageDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.fromarray((self.data[idx] * 255).astype('uint8'))  # Convert to PIL Image
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label


# In[7]:


import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from scipy.stats import entropy

# Function to calculate Inception Score
def calculate_inception_score(images, model, batch_size=32, splits=1):
    """
    Calculate Inception Score for a given set of images.

    Args:
        images (torch.Tensor): A tensor containing the images.
        model (torch.nn.Module): Inception model.
        batch_size (int, optional): Batch size for feeding images to the model. Default is 32.
        splits (int, optional): Number of splits for calculating score. Default is 1.

    Returns:
        float: Inception Score.
    """
    model.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size].to(device)
            pred = F.softmax(model(batch), dim=1)
            preds.append(pred.cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        scores = []
        for i in range(splits):
            part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
            kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))

        return np.mean(scores), np.std(scores)

# Function to prepare images for Inception model
def prepare_inception(imgs):
    """
    Prepare images for Inception model.

    Args:
        imgs (torch.Tensor): A tensor containing the images.

    Returns:
        torch.Tensor: Preprocessed images.
    """
    imgs = F.interpolate(imgs, size=(299, 299), mode='bilinear', align_corners=False)
    return imgs


# In[8]:


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

cuda = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size=64
n=5

# Assuming target_size is (128, 128)
transform = transforms.Compose([
    transforms.Resize(target_size),
    transforms.ToTensor(),
])

# Create instances of the dataset and dataloaders
train_dataset = SignLanguageDataset(X_train, Y_train, transform=transform)
test_dataset = SignLanguageDataset(X_test, Y_test, transform=transform)

sign_language_train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Hyperparameters
n_epochs = 2000
batch_size = 64
lr = 0.0002
b1 = 0.5
b2 = 0.999
n_cpu = 8
latent_dim = 100
img_size = 128
channels = 3
n_critic = 5
clip_value = 0.01
sample_interval = 400
evaluation_interval = 10  # Adjust the interval based on your needs


img_shape = (channels, img_size, img_size)

import torchvision.models as models

# Define the Inception model
inception_model = models.inception_v3(pretrained=True, aux_logits=True)
inception_model.eval()


# Move the model to the GPU if available
if cuda:
    inception_model.cuda()

    
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
        img = 0.5 * img + 0.5  # Adjust pixel values to the range [0, 1]
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

# Before the training loop
d_losses = []
g_losses = []

# Optional: Evaluation metrics
inception_scores = []



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
            
            # Store losses
            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())
            
            # Print losses
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, n_epochs, i, len(sign_language_train_dataloader), d_loss.item(), g_loss.item())
            )

            # Save generated images at specified intervals
            if batches_done % sample_interval == 0:
                save_image(fake_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=False)


            batches_done += n_critic
            
        # Evaluate Inception Score and FID (Optional)
        # if i % evaluation_interval == 0: For every batch
        if epoch % evaluation_interval == 0 and i == 0:  # Evaluate only at the beginning of an epoch
            generator.eval()

            # Use the already generated fake images
            eval_imgs = fake_imgs.detach()

            # Prepare images for Inception model
            eval_imgs = prepare_inception(eval_imgs)

            # Calculate Inception Score
            inception_mean, inception_std = calculate_inception_score(eval_imgs, inception_model, batch_size=32, splits=10)

            # Append Inception Score to the list
            inception_scores.append(inception_mean)
            
            # Print or store the Inception Score
            print(f"[Epoch {epoch}/{n_epochs}] [Batch {i}/{len(sign_language_train_dataloader)}] Inception Score: {inception_mean} +/- {inception_std}")

            # Save generated images at specified intervals
            save_image(eval_imgs.data[:25], f"images/{epoch}_{i}_fake.png", nrow=5, normalize=False)

            # Visualize generated images
            plt.figure(figsize=(10, 6))
            for j in range(1, 6):
                ax = plt.subplot(1, 5, j)
                plt.imshow(eval_imgs[j - 1].cpu().detach().numpy().transpose(1, 2, 0))
                plt.axis('off')
            plt.show()


    # Switch discriminator back to training mode
    discriminator.train()
    
    # Print overall adversarial loss
    print("Overall Adversarial Loss: {:.4f}".format(np.mean(d_losses)))

    # ...
    plt.figure(figsize=(10, 6))
    for i in range(1, n + 1):
        ax = plt.subplot(1, n, i)
        noise = np.random.normal(0, 1, (1, latent_dim))
        # Switch generator to evaluation mode
        generator.eval()
        generated_img = generator(Variable(torch.FloatTensor(noise).cuda() if cuda else torch.FloatTensor(noise))).cpu().detach().numpy()
        # Switch generator back to training mode
        generator.train()

        # Ensure the shape is compatible with RGB images
        generated_img = generator(z).cpu().detach().numpy()[0].transpose(1, 2, 0)

        plt.imshow(generated_img)
        plt.axis('off')

    plt.show()




# In[13]:


# Plot losses
plt.plot(d_losses, label="Discriminator Loss")
plt.plot(g_losses, label="Generator Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()


# In[14]:


# Plot the Inception Scores
plt.figure(figsize=(10, 6))
plt.plot(range(0, n_epochs, evaluation_interval), inception_scores, label='Inception Score')
plt.xlabel('Epoch')
plt.ylabel('Inception Score')
plt.title('Inception Score over Epochs')
plt.legend()
plt.show()


# In[15]:


import matplotlib.pyplot as plt

def evaluate_model(generator, discriminator, test_dataloader, latent_dim, cuda=True):
    generator.eval()
    discriminator.eval()

    correct = 0
    total = 0

    # Lists to store discriminator and generator losses during evaluation
    eval_d_losses = []
    eval_g_losses = []

    with torch.no_grad():
        for i, (imgs, _) in enumerate(test_dataloader):
            real_imgs = Variable(imgs.type(Tensor))

            # Discriminator
            real_validity = discriminator(real_imgs)
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim))))
            fake_imgs = generator(z)
            fake_validity = discriminator(fake_imgs)

            # Generator loss
            g_loss = -torch.mean(fake_validity)
            eval_g_losses.append(g_loss.item())

            # Discriminator loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
            eval_d_losses.append(d_loss.item())

            # Accuracy calculation
            predictions = (fake_validity < 0.5).float()
            correct += (predictions == 0).sum().item()
            total += imgs.size(0)

    accuracy = correct / total

    return accuracy, eval_d_losses, eval_g_losses

# After the training loop
accuracy, eval_d_losses, eval_g_losses = evaluate_model(generator, discriminator, test_dataloader, latent_dim, cuda)

# Print accuracy
print("Test Accuracy: {:.2%}".format(accuracy))

# Plot discriminator and generator losses during evaluation
plt.figure(figsize=(10, 5))
plt.plot(eval_d_losses, label='Discriminator Loss')
plt.plot(eval_g_losses, label='Generator Loss')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.legend()
plt.title('Discriminator and Generator Loss during Evaluation')
plt.show()

# Show Inception Score
generator.eval()
inception_score_mean, inception_score_std = calculate_inception_score(generator(z), inception_model, batch_size=32, splits=10)
print(f"Inception Score: {inception_score_mean} +/- {inception_score_std}")


# In[ ]:





# In[ ]:




