import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from CNN import StarDetector
from Data_generation import generate_examples

class CircleDataset(Dataset):
    def __init__(self, noise_level=0.5, img_size=100, num_samples=1000,):
        self.noise_level = noise_level
        self.img_size = img_size
        self.num_samples = num_samples
        self.transform = transforms.Normalize(mean=[0.5], std=[0.5])

        
    
        self.data_generator = generate_examples(
            noise_level=noise_level,
            img_size=img_size,
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image, target = next(self.data_generator)

        # Apply the transformations
        
        if self.transform:
            image = np.expand_dims(np.asarray(image), axis=0)
            image = torch.from_numpy(np.array(image, dtype=np.float32))
            target = torch.from_numpy(np.array(np.asarray(target), dtype=np.float32))
            image = self.transform(image)
        return image, target
    

# Define your dataset
train_dataset = CircleDataset(
    noise_level=0.5,
    img_size=224,
    num_samples=200000, # Adjust the number of samples as needed
)

# Define your DataLoader
batch_size = 64  # Adjust the batch size as needed
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define your test dataset
test_dataset = CircleDataset(
    noise_level=0.5,
    img_size=224,
    num_samples=500 # Adjust the number of test samples as needed
)
# Define your test DataLoader
test_batch_size = 32  # Adjust the batch size for testing as needed
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

model = StarDetector()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = nn.DataParallel(model)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion1 = nn.MSELoss()


from tqdm import tqdm  # Import tqdm for progress bar

# ... (your previous imports and code)

# Training loop with tqdm
num_epochs = 10


for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    # Use tqdm to create a progress bar for the training loader
    with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as tqdm_bar:
        for inputs, targets in tqdm_bar:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion1(outputs, targets/200)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Update the tqdm description with the current loss
            tqdm_bar.set_postfix(loss=loss.item(), total_loss=total_loss / (tqdm_bar.n + 1))

    # Print the average loss at the end of each epoch
    average_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss}")

    torch.save(model.state_dict(), 'circle_detection.pth')