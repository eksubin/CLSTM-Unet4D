import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from monai.losses import DiceLoss
from monai.data import DataLoader, create_test_image_3d
from clstm_unet3d import UNet3DWithTime

# Custom Dataset for 3D Image Generation
class Custom3DDataset(Dataset):
    def __init__(self, num_samples, time_steps, height, width, depth):
        self.num_samples = num_samples
        self.time_steps = time_steps
        self.height = height
        self.width = width
        self.depth = depth
        self.data_cache = {}
        self.generate_data()  # Generate and cache data

    def generate_data(self):
        for i in range(self.num_samples):
            # Generate a unique key for caching
            key = f'sample_{i}'
            if key not in self.data_cache:
                data_volume = torch.zeros((self.time_steps, 1, self.height, self.width, self.depth))
                data_labels = torch.zeros((self.time_steps, 1, self.height, self.width, self.depth))
                
                for t in range(self.time_steps):
                    test_image, test_seg = create_test_image_3d(self.height, self.width, self.depth, num_objs=12, rad_max=15)
                    data_volume[t] = torch.tensor(test_image)
                    data_labels[t] = torch.tensor(test_seg)

                # Cache the data
                self.data_cache[key] = (data_volume, data_labels)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        key = f'sample_{idx}'
        return self.data_cache[key]

def main():
    device = torch.device("cuda:0")
    # Set constants for the model
    BATCH_SIZE = 2
    TIME_STEPS = 5
    CHANNELS = 1
    DEPTH = 32
    HEIGHT = 128
    WIDTH = 128
    N_CLASSES = 1
    NUM_EPOCHS = 100
    NUM_SAMPLES = 20  # Number of data samples to generate

    # Create dataset and dataloader
    dataset = Custom3DDataset(num_samples=NUM_SAMPLES, time_steps=TIME_STEPS, height=HEIGHT, width=WIDTH, depth=DEPTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Print the model shapes for the first batch
    train_volumes, train_labels = next(iter(dataloader))
    print(f'Train and test dimensions: {train_labels.shape}, {train_volumes.shape}')

    # Instantiate the model
    model = UNet3DWithTime(n_channels=CHANNELS, n_classes=N_CLASSES).to(device)

    # Define loss function and optimizer
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    min_loss = float('inf')  # Initialize min_loss to infinity
    best_model_path = 'unet3d_with_time_best.pth'  # Path to save the best model

    for epoch in range(NUM_EPOCHS):
        model.train()
        for batch in dataloader:
            optimizer.zero_grad()
            train_volumes, train_labels = batch

            print(train_labels.shape, train_volumes.shape)
            # Move to the device
            train_volumes, train_labels = train_volumes.to(device), train_labels.to(device)

            # Forward pass
            output = model(train_volumes)
            
            # Compute the loss
            loss = criterion(output, train_labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}")

            # Save the model if the current loss is lower than min_loss
            if loss.item() < min_loss:
                min_loss = loss.item()
                torch.save(model.state_dict(), best_model_path)
                print(f'Saved best model with loss: {min_loss:.4f}')

    print(f'Training complete. Best model saved at: {best_model_path}')

if __name__ == "__main__":
    print('Model training started')
    main()