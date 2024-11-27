# %%
# Import libraries
import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import mlflow
from torch.nn import DataParallel

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.config import print_config
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    Resized,
    ScaleIntensityd,
    SpatialPadd,
    RandSpatialCropSamplesd,
    LambdaD,
    ConcatItemsd
)
from monai.metrics import DiceMetric
from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)
from model import UNet3DWithTime
#print_config()

# %%
# Setup parameters and device
lr = 1e-4
max_iterations = 10000
eval_num = 10
experiment_name = "clstm_test"
epochs = 10000
dropout = 0.1
train_dir = "../Data/VoiceUsers/Train/Train/"
val_dir = "../Data/VoiceUsers/Val/Nasal25/"
note = "voice_users"
logdir = "./logs"

device = torch.device("cuda:1")
print(f"Device used for processing {device}")

# %%
def create_multichannel_datalist(image_filenames,label_filenames, stack_size=5):
    print(len(image_filenames))
    print(len(label_filenames))
    datalist = []
    end = len(image_filenames) - stack_size + 1
    start = 0
    while start <= end:
        img_dict = {}
        for j in range(0,stack_size):
            img_dict[f"image{j}"] = str(image_filenames[start + j])
        for j in range(0,stack_size):
            img_dict[f"label{j}"] = str(label_filenames[start + j])
        datalist.append(img_dict)
        start += stack_size
    return datalist

# Convert train and validation images into lists with locations
train_nrrd_files = sorted([os.path.join(train_dir, f) for f in os.listdir(
    train_dir) if f.endswith(".nrrd") and not f.endswith(".seg.nrrd")])
train_seg_nrrd_files = sorted([os.path.join(train_dir, f)
                              for f in os.listdir(train_dir) if f.endswith(".seg.nrrd")])

val_nrrd_files = sorted([os.path.join(val_dir, f) for f in os.listdir(
    val_dir) if f.endswith(".nrrd") and not f.endswith(".seg.nrrd")])
val_seg_nrrd_files = sorted([os.path.join(val_dir, f)
                            for f in os.listdir(val_dir) if f.endswith(".seg.nrrd")])

train_datalist = create_multichannel_datalist(train_nrrd_files, train_seg_nrrd_files)
validation_datalist = create_multichannel_datalist(val_nrrd_files, val_seg_nrrd_files,)
print(f" Trian datalist setup {train_datalist[0]}")

# %%
# Define transforms for training and validation
def binarize_label(label):
    return (label > 0).astype(label.dtype)


def threshold_image(image):
    return np.where(image < 0.08, 0, image)


train_transforms = Compose([
    LoadImaged(keys=["image0", "image1", "image2", "image3", "image4", "label0","label1","label2","label3","label4"]),
    EnsureChannelFirstd(keys=["image0", "image1", "image2", "image3", "image4", "label0","label1","label2","label3","label4"]),
    ConcatItemsd(keys=["image0", "image1", "image2",
                     "image3", "image4"], name="image"),
    ConcatItemsd(keys=["label0","label1","label2","label3","label4"], name="label"),                 
    ScaleIntensityd(keys=["image"], minv=0, maxv=1),
    LambdaD(keys="label", func=binarize_label),
    LambdaD(keys="image", func=threshold_image),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    SpatialPadd(keys=["image", "label"], spatial_size=(64, 64, 64)),
    RandSpatialCropSamplesd(keys=["image", "label"], roi_size=(
        64, 64, 64), random_size=False, num_samples=4),
    RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.10),
    RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.10),
    RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.10),
    RandRotate90d(keys=["image", "label"], prob=0.10, max_k=3),
    RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.50),
    ToTensord(keys=["image", "label"]),
])

val_transforms = Compose([
    LoadImaged(keys=["image0", "image1", "image2", "image3",
               "image4", "label0", "label1", "label2", "label3", "label4"]),
    EnsureChannelFirstd(keys=["image0", "image1", "image2", "image3",
                        "image4", "label0", "label1", "label2", "label3", "label4"]),
    ConcatItemsd(keys=["image0", "image1", "image2",
                       "image3", "image4"], name="image"),
    ConcatItemsd(keys=["label0", "label1", "label2",
                 "label3", "label4"], name="label"),
    ScaleIntensityd(keys=["image"], minv=0, maxv=1),
    LambdaD(keys="label", func=binarize_label),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    SpatialPadd(keys=["image", "label"], spatial_size=(64, 64, 64)),
    RandSpatialCropSamplesd(keys=["image", "label"], roi_size=(
        64, 64, 64), random_size=False, num_samples=4),
    ToTensord(keys=["image", "label"]),
])

# %%
# Create DataLoaders for training and validation
train_ds = CacheDataset(data=train_datalist, transform=train_transforms,
                        cache_num=24, cache_rate=1.0, num_workers=2)
train_loader = DataLoader(train_ds, batch_size=1,
                          shuffle=True, num_workers=4, pin_memory=True)

val_ds = CacheDataset(data=validation_datalist, transform=val_transforms,
                      cache_num=6, cache_rate=1.0, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=1,
                        shuffle=False, num_workers=4, pin_memory=True)

# %%
# Define and setup the model
model = UNet3DWithTime(n_channels=1, n_classes=1).to(device)
model = model.to(device)

# model parallelization
# if torch.cuda.device_count() > 1:

#     model = DataParallel(model)
#     print(f"###### Using data parallism {torch.cuda.device_count()}")

# %%
# Setup loss function, optimizer, and metrics
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
torch.backends.cudnn.benchmark = True
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

post_label = AsDiscrete(to_onehot=2)
post_pred = AsDiscrete(argmax=True, to_onehot=2)
dice_metric = DiceMetric(include_background=True,
                         reduction="mean", get_not_nans=False)
global_step = 0
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []

# %%
# %% Define the validation component


def validation(epoch_iterator_val, dice_val_best):
    model.eval().to("cuda:1")
    dice_vals = []

    with torch.no_grad():
        for _step, batch in enumerate(epoch_iterator_val):
            val_inputs, val_labels = (
                batch["image"].unsqueeze(2).to("cuda:1"), 
                batch["label"].unsqueeze(2).to("cuda:1")
            )
            print("Validation shape ", val_inputs.shape, val_labels.shape)
            if val_labels.shape[1] != 1:
                # Assuming y originally has more than 1 channel, and you need to reduce it
                # Simplify to first channel, modify as needed
                val_labels = val_labels[:, 0, ...]
                val_labels = val_labels.unsqueeze(1)
            
            val_outputs = model(val_inputs)
            
            if val_outputs.shape[1] != 1:
                # Assuming y originally has more than 1 channel, and you need to reduce it
                # Simplify to first channel, modify as needed
                val_outputs = val_outputs[:, 0, ...]
                val_outputs = val_outputs.unsqueeze(1)

            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [post_label(
                val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [post_pred(val_pred_tensor)
                                  for val_pred_tensor in val_outputs_list]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            dice = dice_metric.aggregate().item()
            dice_vals.append(dice)
            epoch_iterator_val.set_description(
                "Validate (%d / %d Steps) (dice=%2.5f)" % (global_step, 10.0, dice))

        dice_metric.reset()

    mean_dice_val = np.mean(dice_vals)
    if mean_dice_val > dice_val_best and mean_dice_val != 1:
        print(f"validation output shapes {val_outputs.shape}")
        image_slice = val_outputs[0, 0, 0, 15, :, :].cpu().numpy()
        image_slice = (image_slice * 255).astype(np.uint8)
        image = Image.fromarray(image_slice)
        image_path = os.path.join(
            logdir, str(mean_dice_val) + experiment_name+ "_output_slice.png")
        image.save(image_path)
        mlflow.log_artifact(image_path)


       # Convert Pillow image to a matplotlib figure
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray')
        ax.axis('off')  # Turn off the axis

        # Save the figure to a temporary file and log it as an MLflow figure
        figure_path = os.path.join(logdir, experiment_name +
                                "_output_slice_figure.png")
        fig.savefig(figure_path, bbox_inches='tight', pad_inches=0)

        # Log the figure as an MLflow figure
        mlflow.log_artifact(figure_path)

    # Log validation dice score
    mlflow.log_metric('val_dice', mean_dice_val, step=global_step)

    return mean_dice_val
    #%% Define training function
from PIL import Image

def train(global_step, train_loader, dice_val_best, global_step_best):
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x = batch["image"].unsqueeze(2).to(device)
        y = batch["label"].unsqueeze(2).to(device)
        
        if y.shape[1] != 1:
            # Assuming y originally has more than 1 channel, and you need to reduce it
            # Simplify to first channel, modify as needed
            y = y[:, 0, ...]
            y = y.unsqueeze(1)
        
        logit_map = model(x)
        if logit_map.shape[1] != 1:
            # Assuming y originally has more than 1 channel, and you need to reduce it
            # Simplify to first channel, modify as needed
            logit_map = logit_map[:, 0, ...]
            logit_map = logit_map.unsqueeze(1)
        print("input shape:", x.shape, y.shape, logit_map.shape)
        
        loss = loss_function(logit_map, y)
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        epoch_iterator.set_description(
            "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss))

        # Log metric loss
        mlflow.log_metric('train_loss', loss.item(),
                          step=global_step)  # Log training loss

        if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:
            epoch_iterator_val = tqdm(
                val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
            dice_val = validation(epoch_iterator_val, dice_val_best)

            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            if dice_val > dice_val_best and dice_val != 1:
                dice_val_best = dice_val
                global_step_best = global_step
                mlflow.log_metric('Current_step', global_step_best)
                # Log best dice value
                mlflow.log_metric('dice_val_best', dice_val_best)
                print(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {} Step {}".format(
                        dice_val_best, dice_val, global_step))
                torch.save(model.state_dict(),
                           os.path.join(logdir, experiment_name + note +  ".pth"))
            else:
                print(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {} Step {}".format(
                        dice_val_best, dice_val, global_step
                    )
                )

        global_step += 1
    return global_step, dice_val_best, global_step_best
#%%
    # Run the program
while global_step < max_iterations:
    global_step, dice_val_best, global_step_best = train(
        global_step, train_loader, dice_val_best, global_step_best)

# Load the best model state
model.load_state_dict(torch.load(
    os.path.join(logdir, experiment_name + ".pth")))
    