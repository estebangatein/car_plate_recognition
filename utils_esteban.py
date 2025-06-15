import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import torch.nn.functional as F


###### FROM BASELINE MODEL ######
CHARS = [
    "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学",
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S',
    'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
]

# needed for the Seq2Seq model
SPECIAL = ['<PAD>', '<BOS>', '<EOS>']
VOCAB = SPECIAL + sorted(set(CHARS))
char2idx = {c: i for i, c in enumerate(VOCAB)}
idx2char = {i: c for c, i in char2idx.items()}

PAD_IDX = char2idx['<PAD>']
BOS_IDX = char2idx['<BOS>']
EOS_IDX = char2idx['<EOS>']

class BoundingBoxCNN(nn.Module):
    def __init__(self):
        super(BoundingBoxCNN, self).__init__()
        
        # convolutions
        self.features = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d((1, 1))
        )

        # fully connected layers
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x
    
class BaselineData(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        path = os.path.join(self.image_dir, filename)

        # image reading
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        # bounding box from file name
        parts = filename.split('-')
        bbox_part = parts[2]
        x1y1, x2y2 = bbox_part.split('_')
        x1, y1 = map(int, x1y1.split('&'))
        x2, y2 = map(int, x2y2.split('&'))

        _, img_height, img_width = image.shape
        
        # normalize the bounding box
        x1 = x1 / img_width
        x2 = x2 / img_width
        y1 = y1 / img_height
        y2 = y2 / img_height

        bbox = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)

        # decodes the plate
        plate_raw = parts[4]
        plate_text = decode_plate(plate_raw)
        
        return image, plate_text, bbox
    

class LicensePlateSeq2Seq(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, max_len=10):
        super().__init__()
        self.max_len = max_len

        # CNNs as encoders
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 24x72
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 12x36
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))  # 1x1
        )
        self.flatten = nn.Flatten()
        self.fc_enc = nn.Linear(256, hidden_dim)

        # LSTM for the text 
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=1, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, img, target_seq=None, teacher_forcing=True):
        # encoder
        feat = self.encoder(img)              
        feat = self.flatten(feat)
        # initial states for the LSTM
        h0 = torch.tanh(self.fc_enc(feat))    
        h0 = h0.unsqueeze(0)                  
        c0 = torch.zeros_like(h0)

        B = img.size(0)
        outputs = []
        input_token = torch.full((B, 1), BOS_IDX, dtype=torch.long, device=img.device)  # [B, 1]

        # generates the tokens through LSTM
        for t in range(self.max_len):
            embed = self.embedding(input_token)  
            out, (h0, c0) = self.lstm(embed, (h0, c0))
            logits = self.output(out.squeeze(1))  
            outputs.append(logits.unsqueeze(1))   

            # set to true in training to help, uses the true sequence
            if teacher_forcing:
                input_token = target_seq[:, t].unsqueeze(1)  
            else:
                input_token = logits.argmax(1).unsqueeze(1) # only for inference

        return torch.cat(outputs, dim=1)


##### FROM THE PAPER-BASED MODEL #####
# utils for decoding the labels
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

# decodes the plate from the file name
def decode_plate(label_str):
    indices = list(map(int, label_str.split('_')))
    province = provinces[indices[0]]
    alphabet = alphabets[indices[1]]
    ad = ''
    for i in range(2, len(indices)):
        ad += ads[indices[i]]

    return province + alphabet + ad

# torch dataset
class Data_Yolo(Dataset):
    def __init__(self, image_dir, transform=None, img_size=(640, 640)):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        self.transform = transform
        self.img_size = img_size  # (H, W)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        path = os.path.join(self.image_dir, filename)

        # load image
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # needed to scale the bounding box
        original_h, original_w = image.shape[:2]

        # bounding box from filename
        parts = filename.split('-')
        bbox_part = parts[2]
        x1y1, x2y2 = bbox_part.split('_')
        x1, y1 = map(int, x1y1.split('&'))
        x2, y2 = map(int, x2y2.split('&'))

        # resize image to 640x640 using + intensity normalization
        resized_image = cv2.resize(image, self.img_size[::-1])
        image_tensor = torch.tensor(resized_image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        # adjust bbox to resized scale
        scale_x = self.img_size[1] / original_w
        scale_y = self.img_size[0] / original_h

        x1_resized = x1 * scale_x / self.img_size[1]
        x2_resized = x2 * scale_x / self.img_size[1]
        y1_resized = y1 * scale_y / self.img_size[0]
        y2_resized = y2 * scale_y / self.img_size[0]

        bbox = torch.tensor([x1_resized, y1_resized, x2_resized, y2_resized], dtype=torch.float32)

        # plate text
        plate_raw = parts[4]
        plate_text = decode_plate(plate_raw)

        return image_tensor, plate_text, bbox
    

class Data_PDLPR(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        path = os.path.join(self.image_dir, filename)
    
        # load image
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
        # bbox from filename
        parts = filename.split('-')
        bbox_part = parts[2]
        x1y1, x2y2 = bbox_part.split('_')
        x1, y1 = map(int, x1y1.split('&'))
        x2, y2 = map(int, x2y2.split('&'))
    
        # crop given the plate bbox
        h, w = image.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        cropped = image[y1:y2, x1:x2]
    
        # resize as the paper
        cropped = cv2.resize(cropped, (144, 48))
        image_tensor = (torch.tensor(cropped, dtype=torch.float32).permute(2, 0, 1) / 255.0 - 0.5) / 0.5 # between -1 and 1

        # plate text
        plate_raw = parts[4]
        plate_text = decode_plate(plate_raw)
    
        return image_tensor, plate_text


# focus block
class Focus(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels * 4, out_channels, 3, 1, 1)

    def forward(self, x):
        return self.conv(torch.cat([
            x[..., ::2, ::2],
            x[..., ::2, 1::2],
            x[..., 1::2, ::2],
            x[..., 1::2, 1::2]
        ], dim=1))


# convolutional block
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# residual block
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(ch, ch),
            ConvBlock(ch, ch)
        )
        self.bn = nn.BatchNorm2d(ch)

    def forward(self, x):
        return self.bn(x + self.block(x))


# IGFE
class IGFE(nn.Module):
    def __init__(self):
        super().__init__()
        self.focus = Focus(3, 64)
        self.down1 = ConvBlock(64, 128, s=2)
        self.res1 = nn.Sequential(ResBlock(128), ResBlock(128))
        self.down2 = ConvBlock(128, 256, s=2)
        self.res2 = nn.Sequential(ResBlock(256), ResBlock(256))
        self.conv_out = nn.Conv2d(256, 512, kernel_size=1)

    def forward(self, x):
        x = self.focus(x)
        x = self.down1(x)
        x = self.res1(x)
        x = self.down2(x)
        x = self.res2(x)
        x = torch.clamp(x, -10, 10)
        return self.conv_out(x)  # (B, 512, 6, 18)


# encoder unit
class EncoderUnit(nn.Module):
    def __init__(self, d_model=512, nhead=8):
        super().__init__()
        self.cnn1 = nn.Conv1d(d_model, 1024, kernel_size=1)
        self.mha = nn.MultiheadAttention(embed_dim=1024, num_heads=nhead, batch_first=True)
        self.cnn2 = nn.Conv1d(1024, d_model, kernel_size=1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x_cnn = x.transpose(1, 2)              
        x_mha_in = F.relu(self.cnn1(x_cnn)).transpose(1, 2)
        attn_out, _ = self.mha(x_mha_in, x_mha_in, x_mha_in) 
        x_proj = self.cnn2(attn_out.transpose(1, 2)).transpose(1, 2)
        return self.norm(x + x_proj)


# full encoder
class PDLPR_Encoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=3):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, 108, d_model))
        self.dropout = nn.Dropout(p=0.1)
        self.layers = nn.Sequential(*[
            EncoderUnit(d_model=d_model, nhead=nhead) for _ in range(num_layers)
        ])

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)  # (B, 108, 512)
        x = self.dropout(x + self.pos_embed[:, :x.size(1), :])
        return self.layers(x)  # (B, 108, 512)


# FFN block (non-linear transformation)
class FeedForward(nn.Module):
    def __init__(self, d_model=512, hidden_dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(x + self.net(x))


# decoder unit
class DecoderUnit(nn.Module):
    def __init__(self, d_model=512, nhead=8):
        super().__init__()
        self.masked_mha = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        self.encoder_proj = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=1)
        )

        self.cross_mha = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model)

    def forward(self, tgt, memory, tgt_mask=None):
        mha_out1, _ = self.masked_mha(tgt, tgt, tgt, attn_mask=tgt_mask)
        x = self.norm1(tgt + mha_out1)

        mem_proj = memory.transpose(1, 2)
        mem_proj = self.encoder_proj(mem_proj).transpose(1, 2)

        mha_out2, _ = self.cross_mha(x, mem_proj, mem_proj)
        x = self.norm2(x + mha_out2)

        return self.ffn(x)


# full parallel decoder
class ParallelDecoder(nn.Module):
    def __init__(self, d_model=512, num_classes=92, nhead=8, num_layers=3, max_seq_len=18):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        self.dropout = nn.Dropout(0.1)

        self.decoder_layers = nn.ModuleList([
            DecoderUnit(d_model, nhead) for _ in range(num_layers)
        ])

        self.out_proj = nn.Linear(d_model, num_classes)

    def forward(self, encoder_out):
        B = encoder_out.size(0)
        T = 18
        device = encoder_out.device

        tgt = torch.zeros(B, T, encoder_out.size(2), device=device) + self.pos_embed[:, :T, :]
        tgt = self.dropout(tgt)

        mask = torch.triu(torch.ones(T, T, device=device) * float('-inf'), diagonal=1)

        for layer in self.decoder_layers:
            tgt = layer(tgt, encoder_out, tgt_mask=mask)

        return self.out_proj(tgt)  # (B, T, num_classes)


# final PDLPRModel
class PDLPRModel(nn.Module):
    def __init__(self, num_classes=92):
        super().__init__()
        self.igfe = IGFE()
        self.encoder = PDLPR_Encoder()
        self.decoder = ParallelDecoder(num_classes=num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.igfe(x)           # (B, 512, 6, 18)
        x = self.encoder(x)        # (B, 108, 512)
        x = self.decoder(x)        # (B, 18, num_classes)
        return x