import torch
from torch import nn

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def normal(feat, eps=1e-5):
    feat_mean, feat_std= calc_mean_std(feat, eps)
    normalized=(feat-feat_mean)/feat_std
    return normalized 

class Network(nn.Module):    
    def __init__(self, source_encoder, target_encoder, TR_module, decoder, vgg):

        super().__init__()
        enc_layers = list(vgg.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1
        
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

        self.mse_loss = nn.MSELoss()
        self.source_encoder = source_encoder
        self.target_encoder = target_encoder
        self.TR_module = TR_module
        self.decoder = decoder

    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def calc_content_loss(self, input, target):
      assert (input.size() == target.size())
      assert (target.requires_grad is False)
      return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)
    
    def forward(self, source_imgs, target_imgs):
        ### Feature Extractor ###
        # Patch embedding
        source_feats = self.source_encoder(source_imgs)
        target_feats = self.target_encoder(target_imgs)
        hs = self.TR_module(source_feats, target_feats)

        # Decoder
        Ics = self.decoder(hs) 
        
        ### Perceptual features ###
        source_vgg = self.encode_with_intermediate(source_imgs)
        target_vgg = self.encode_with_intermediate(target_imgs)
        Ics_vgg = self.encode_with_intermediate(Ics)

        # Content loss
        loss_c = self.calc_content_loss(Ics_vgg[-1], source_vgg[-1]) + self.calc_content_loss(Ics_vgg[-2], source_vgg[-2])
        # Style loss
        loss_s = self.calc_style_loss(Ics_vgg[0], target_vgg[0])
        for i in range(1, 5):
            loss_s += self.calc_style_loss(Ics_vgg[i], target_vgg[i])
        
        return Ics,  loss_c, loss_s
