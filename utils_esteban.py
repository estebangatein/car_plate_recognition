import torch
import torch.nn as nn


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

# image downsampling (better than pooling)
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

# convolution sequence block
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_ch)  
        self.act = nn.LeakyReLU(0.1, inplace=False)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# residual blocks
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(ch, ch),
            ConvBlock(ch, ch)
        )
        self.bn = nn.BatchNorm2d(ch) 

    def forward(self, x):
        out = self.block(x)
        return self.bn(x + out)


# Image Global Feature Extractor block (combines the previous blocks)
class IGFE(nn.Module):
    def __init__(self):
        super().__init__()
        self.focus = Focus(3, 64)
        self.down1 = ConvBlock(64, 128, s=2)
        self.down2 = ConvBlock(128, 256, s=2)
        self.res = nn.Sequential(
            ResBlock(256),
            ResBlock(256),
            ResBlock(256),
            ResBlock(256)
        )
        self.conv_out = nn.Conv2d(256, 512, 1)

    def forward(self, x):
        x = self.focus(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.res(x)
        x = torch.clamp(x, -10, 10)  # safety clamp
        x = self.conv_out(x)
        return x

# transformer encoding from image
class TransformerEncoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=3):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, 108, d_model)) 
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)  
        x = self.dropout(x + self.pos_embed)
        x = self.encoder(x)
        return x



# prediction block (decodes the text)
class ParallelDecoder(nn.Module):
    def __init__(self, d_model=512, num_classes=92):
        super().__init__()
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        return self.head(x)

# full model (Parallel Deep-Learning License Plate Recognition)
class PDLPRModel(nn.Module):
    def __init__(self, num_classes=92):
        super().__init__()
        self.igfe = IGFE()
        self.encoder = TransformerEncoder()
        self.decoder = ParallelDecoder(num_classes=num_classes)

        self._init_weights()


    def _init_weights(self): # needed because of unstable training
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    


    def forward(self, x):
        x = self.igfe(x)
        x = self.encoder(x)
        x = self.decoder(x)

        return x


