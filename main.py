
import os
import torch
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import csv
import sys
from thop import profile
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import pickle
import joblib
import random

gpu_device = 0
torch.cuda.set_device(gpu_device)
print("Available CUDA Device: ", torch.cuda.get_device_name(0))


# 自定义的数据集类，用于加载和处理图像数据及其对应的标签
class CustomDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path, label = self.data_list[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)  # Convert image to tensor
        return image, torch.tensor(label, dtype=torch.float)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

data_dirs = [
    "./data",
]

# Collect all image paths and labels
all_image_paths = []
all_labels = []

for data_dir in data_dirs:
    for folder_name in sorted(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, folder_name)
        if os.path.isdir(folder_path):
            images = sorted(os.listdir(folder_path))
            image_paths = [os.path.join(folder_path, img) for img in images]
            all_image_paths.extend(image_paths)

            # Extract labels
            x, y, z, angle = folder_name.strip("()").split(',')
            x = float(x.replace('cm', '').strip())
            y = float(y.replace('cm', '').strip())
            z = float(z.replace('cm', '').strip())
            angle = float(angle.strip())
            labels = [[x, y, z, angle]] * len(image_paths)  # Replicate label for each image
            all_labels.extend(labels)

# Combine image paths and labels, then shuffle
data = list(zip(all_image_paths, all_labels))
random.seed(42)
random.shuffle(data)

# Split data into training (70%), validation (20%), and test (10%)
N = len(data)
train_size = int(0.7 * N)  # 训练数据集
val_size = int(0.2 * N)  # 验证数据集
test_size = N - train_size - val_size  # 测试数据集

train_data = data[:train_size]
val_data = data[train_size:train_size + val_size]
test_data = data[train_size + val_size:]
# Create datasets
train_dataset = CustomDataset(train_data, transform=transform)
valid_dataset = CustomDataset(val_data, transform=transform)
test_dataset = CustomDataset(test_data, transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)


class TransformerAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(TransformerAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        batch_size, num_channels, height, width = x.shape
        x = x.view(batch_size, num_channels, -1).permute(2, 0, 1)
        attn_output, _ = self.multihead_attn(x, x, x)
        attn_output = self.norm(attn_output + x)
        attn_output = attn_output.permute(1, 2, 0).view(batch_size, num_channels, height, width)
        return attn_output


class ResNet50Custom(nn.Module):
    def __init__(self, num_outputs=4):
        super(ResNet50Custom, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)

        for param in self.resnet50.parameters():
            param.requires_grad = False

        num_ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs, num_outputs)
        )

        self.transformer_attention = TransformerAttention(embed_dim=1024, num_heads=8)
        nn.init.kaiming_normal_(self.transformer_attention.multihead_attn.in_proj_weight, mode='fan_out',
                                nonlinearity='relu')
        nn.init.constant_(self.transformer_attention.multihead_attn.in_proj_bias, 0)

    def forward(self, x):
        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)

        x = self.resnet50.layer1(x)
        x = self.resnet50.layer2(x)
        x = self.resnet50.layer3(x)

        x = self.transformer_attention(x)

        x = self.resnet50.layer4(x)
        x = self.resnet50.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet50.fc(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_example = torch.randn(1, 3, 224, 224)
resnet50 = ResNet50Custom(num_outputs=4).to(device)
flops, params = profile(resnet50, inputs=(input_example.to(device),))
print(f"FLOPs: {flops}, Params: {params}")

criterion = nn.MSELoss()
optimizer = optim.Adam(resnet50.parameters(), lr=0.1)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=15)


def validate_model(model, valid_loader, criterion):
    model.eval()
    validation_loss = 0.0
    all_labels = []
    all_outputs = []

    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            validation_loss += loss.item()
            all_labels.append(labels.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())

    validation_loss /= len(valid_loader)
    all_labels = np.concatenate(all_labels, axis=0)
    all_outputs = np.concatenate(all_outputs, axis=0)

    # Using XGBoost Regressor
    xgb_model_x = xgb.XGBRegressor()
    xgb_model_y = xgb.XGBRegressor()
    xgb_model_z = xgb.XGBRegressor()
    xgb_model_angle = xgb.XGBRegressor()

    xgb_model_x.fit(all_outputs, all_labels[:, 0])
    xgb_model_y.fit(all_outputs, all_labels[:, 1])
    xgb_model_z.fit(all_outputs, all_labels[:, 2])
    xgb_model_angle.fit(all_outputs, all_labels[:, 3])

    # Save XGBoost models
    with open('xgboost_model.pkl', 'wb') as f:
        pickle.dump({'x': xgb_model_x, 'y': xgb_model_y, 'z': xgb_model_z, 'angle': xgb_model_angle}, f)
    pred_x = xgb_model_x.predict(all_outputs)
    pred_y = xgb_model_y.predict(all_outputs)
    pred_z = xgb_model_z.predict(all_outputs)
    pred_angle = xgb_model_angle.predict(all_outputs)

    mse_x = mean_squared_error(all_labels[:, 0], pred_x)
    mse_y = mean_squared_error(all_labels[:, 1], pred_y)
    mse_z = mean_squared_error(all_labels[:, 2], pred_z)
    mse_angle = mean_squared_error(all_labels[:, 3], pred_angle)

    rmse_x = np.sqrt(mse_x)
    rmse_y = np.sqrt(mse_y)
    rmse_z = np.sqrt(mse_z)
    rmse_angle = np.sqrt(mse_angle)

    mae_x = mean_absolute_error(all_labels[:, 0], pred_x)
    mae_y = mean_absolute_error(all_labels[:, 1], pred_y)
    mae_z = mean_absolute_error(all_labels[:, 2], pred_z)
    mae_angle = mean_absolute_error(all_labels[:, 3], pred_angle)

    # Combined XYZ
    mse_xyz = (mse_x + mse_y + mse_z)
    rmse_xyz = np.sqrt(mse_x + mse_y + mse_z)
    mae_xyz = (mae_x + mae_y + mae_z)

    # XY coordinates
    mse_xy = (mse_x + mse_y)
    rmse_xy = np.sqrt(mse_x + mse_y)
    mae_xy = (mae_x + mae_y)

    return validation_loss, mse_xyz, rmse_xyz, mae_xyz, mse_xy, rmse_xy, mae_xy, mse_angle, rmse_angle, mae_angle


# 测试、预测剩下10%图像坐标
def evaluate_testset(model, test_loader):
    model.eval()
    error_xy_list = []
    error_angle_list = []

    # Load XGBoost models
    with open('xgboost_model.pkl', 'rb') as f:
        xgb_models = pickle.load(f)
        xgb_model_x = xgb_models['x']
        xgb_model_y = xgb_models['y']
        xgb_model_z = xgb_models['z']
        xgb_model_angle = xgb_models['angle']

    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            all_outputs.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_outputs = np.concatenate(all_outputs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Use XGBoost models to predict
    pred_x = xgb_model_x.predict(all_outputs)
    pred_y = xgb_model_y.predict(all_outputs)
    pred_z = xgb_model_z.predict(all_outputs)
    pred_angle = xgb_model_angle.predict(all_outputs)

    # Compute errors
    for i in range(len(all_labels)):
        true_x, true_y, true_z, true_angle = all_labels[i]
        predicted_x = pred_x[i]
        predicted_y = pred_y[i]
        predicted_angle = pred_angle[i]

        # xyangle定位误差,没有加z误差
        error_xy = np.sqrt((predicted_x - true_x) ** 2 + (predicted_y - true_y) ** 2)
        error_angle = abs(predicted_angle - true_angle)

        error_xy_list.append(error_xy)
        error_angle_list.append(error_angle)

    error_average_xy = sum(error_xy_list) / len(error_xy_list)
    error_average_angle = sum(error_angle_list) / len(error_angle_list)

    return error_average_xy, error_average_angle


def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs=140, batch_size=64):
    dropout_value = model.resnet50.fc[0].p
    patience_value = scheduler.patience

    filename = f"Mdpi-mutil-xgboost-resnet50_log_dropout{dropout_value}_epochs{num_epochs}_patience{patience_value}_batch{batch_size}.csv"
    writer = SummaryWriter('Mdpi-mutil-xgboost-resnet50-runs/training_analysis')
    best_loss = float('inf')

    with open('error_xyangle.csv', 'w', newline='') as error_file, open(filename, 'w', newline='') as file:
        error_writer = csv.writer(error_file)
        error_writer.writerow(['Epoch', 'Error Average XY', 'Error Average Angle'])

        log_writer = csv.writer(file)
        log_writer.writerow(
            ['Epoch', 'Training Loss', 'Validation Loss', 'Validation MSE XYZ', 'Validation RMSE XYZ',
             'Validation MAE XYZ', 'Validation MSE XY', 'Validation RMSE XY', 'Validation MAE XY',
             'Validation MSE angle', 'Validation RMSE angle', 'Validation MAE angle', 'FLOPs', 'Params'])

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            training_loss = running_loss / len(train_loader)

            validation_loss, mse_xyz, rmse_xyz, mae_xyz, mse_xy, rmse_xy, mae_xy, mse_angle, rmse_angle, mae_angle = validate_model(
                model, valid_loader, criterion)

            # Evaluate on test set
            error_average_xy, error_average_angle = evaluate_testset(model, test_loader)
            error_writer.writerow([epoch + 1, error_average_xy, error_average_angle])

            writer.add_scalars('Loss', {'Training': training_loss, 'Validation': validation_loss}, epoch)
            writer.add_scalar('MSE XYZ', mse_xyz, epoch)
            writer.add_scalar('RMSE XYZ', rmse_xyz, epoch)
            writer.add_scalar('MAE XYZ', mae_xyz, epoch)
            writer.add_scalar('MSE XY', mse_xy, epoch)
            writer.add_scalar('RMSE XY', rmse_xy, epoch)
            writer.add_scalar('MAE XY', mae_xy, epoch)
            writer.add_scalar('MSE angle', mse_angle, epoch)
            writer.add_scalar('RMSE angle', rmse_angle, epoch)
            writer.add_scalar('MAE angle', mae_angle, epoch)

            log_writer.writerow(
                [epoch + 1, training_loss, validation_loss, mse_xyz, rmse_xyz, mae_xyz, mse_xy, rmse_xy, mae_xy,
                 mse_angle,
                 rmse_angle, mae_angle, flops, params])
            scheduler.step(validation_loss)

            if validation_loss < best_loss:
                best_loss = validation_loss
                torch.save(model.state_dict(), 'QuanZhong-Mdpi-mutil-xgboost-resnet50.pth')
                torch.save(model, 'Model-Mdpi-mutil-xgboost-resnet50.pth')
                print("Model saved.")

            print(
                f'Epoch {epoch + 1}, Training Loss: {training_loss:.3f}, Validation Loss: {validation_loss:.3f}, MSE XYZ: {mse_xyz:.3f}, RMSE XYZ: {rmse_xyz:.3f}, MAE XYZ: {mae_xyz:.3f}, MSE XY: {mse_xy:.3f}, RMSE XY: {rmse_xy:.3f}, MAE XY: {mae_xy:.3f}, MSE angle:{mse_angle:.3f}, RMSE angle: {rmse_angle:.3f}, MAE angle: {mae_angle:.3f}')

    writer.close()
    print("Training completed.")


train_model(resnet50, train_loader, valid_loader, criterion, optimizer, scheduler)
