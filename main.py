import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from unet import UNet
from caravana_dataset import CaravanaDataset
from torch.utils.tensorboard import SummaryWriter
import os

# Initialisation SummaryWriter
writer = SummaryWriter('runs/U-Net_1')

def dice_coefficient(prediction, target, epsilon=1e-07):
    prediction_copy = prediction.clone()
    prediction_copy[prediction_copy < 0] = 0
    prediction_copy[prediction_copy > 0] = 1
    intersection = abs(torch.sum(prediction_copy * target))
    union = abs(torch.sum(prediction_copy) + torch.sum(target))
    dice = (2. * intersection + epsilon) / (union + epsilon)
    return dice

if __name__ == "__main__":
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 8
    EPOCHS = 2
    DATA_PATH = "C:/Users/mathi/Desktop/U-Net project/data"
    MODEL_SAVE_PATH = "C:/Users/mathi/Desktop/U-Net project/models/unet.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cuda":
        torch.cuda.empty_cache()

    # Créer le dossier models s'il n'existe pas
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    train_dataset = CaravanaDataset(DATA_PATH)
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(train_dataset, [0.8, 0.2], generator=generator)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = UNet(in_channels=3, num_classes=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    # Ajout du modèle à TensorBoard
    dummy_input = torch.randn(1, 3, 512, 512).to(device)
    writer.add_graph(model, dummy_input)

    # Ajouter les hyperparamètres
    hparams = {
        'learning_rate': LEARNING_RATE,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS
    }
    writer.add_hparams(hparams, {})

    print("Début de l'entraînement de U-Net")

    # Boucle d'entraînement
    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_running_loss = 0
        train_running_dc = 0
        
        for idx, img_mask in enumerate(tqdm(train_dataloader)):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)

            y_pred = model(img)
            optimizer.zero_grad()

            loss = criterion(y_pred, mask)
            train_running_loss += loss.item()

            dc = dice_coefficient(y_pred, mask)
            train_running_dc += dc.item()

            loss.backward()
            optimizer.step()

            # Log par batch (optionnel)
            if idx % 10 == 0:
                global_step = epoch * len(train_dataloader) + idx
                writer.add_scalar('Training/Loss_per_batch', loss.item(), global_step)
                writer.add_scalar('Training/Dice_per_batch', dc.item(), global_step)

        train_loss = train_running_loss / (idx + 1)
        train_dc = train_running_dc / (idx + 1)

        # Boucle de validation
        model.eval()
        val_running_loss = 0
        val_running_dc = 0
        
        with torch.no_grad():
            for idx, img_mask in enumerate(tqdm(val_dataloader)):
                img = img_mask[0].float().to(device)
                mask = img_mask[1].float().to(device)

                y_pred = model(img)
                val_loss = criterion(y_pred, mask)
                val_dc = dice_coefficient(y_pred, mask)

                val_running_loss += val_loss.item()
                val_running_dc += val_dc.item()

        val_loss_avg = val_running_loss / (idx + 1)
        val_dc_avg = val_running_dc / (idx + 1)

        # Log des métriques par époque
        writer.add_scalar('Training/Loss_per_epoch', train_loss, epoch)
        writer.add_scalar('Training/Dice_per_epoch', train_dc, epoch)
        writer.add_scalar('Validation/Loss_per_epoch', val_loss_avg, epoch)
        writer.add_scalar('Validation/Dice_per_epoch', val_dc_avg, epoch)

        # Log des images (première image du batch)
        with torch.no_grad():
            sample_img, sample_mask = next(iter(val_dataloader))
            sample_img = sample_img[0:1].float().to(device)
            sample_mask = sample_mask[0:1].float().to(device)
            sample_pred = model(sample_img)
            sample_pred_sigmoid = torch.sigmoid(sample_pred)
            
            writer.add_image('Validation/Original_Image', sample_img[0], epoch)
            writer.add_image('Validation/Ground_Truth_Mask', sample_mask[0], epoch)
            writer.add_image('Validation/Predicted_Mask', sample_pred_sigmoid[0], epoch)

        print("-" * 30)
        print(f"Training Loss EPOCH {epoch + 1}: {train_loss:.4f}")
        print(f"Training DICE EPOCH {epoch + 1}: {train_dc:.4f}")
        print(f"Validation Loss EPOCH {epoch + 1}: {val_loss_avg:.4f}")
        print(f"Validation DICE EPOCH {epoch + 1}: {val_dc_avg:.4f}")
        print("-" * 30)

    # Fermer le writer TensorBoard
    writer.close()
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("Entraînement terminé!")