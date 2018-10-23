class VGG11_BN_FCN(nn.Module):
    def __init__(self):
        super().__init__()
        vgg11_bn = models.vgg11_bn(pretrained=True)
        self.encoder = vgg11_bn.features
        self.decoder = nn.Sequential(nn.Conv2d(in_channels=512,out_channels=256,kernel_size=1),
                                     nn.Conv2d(in_channels=256,out_channels=1,kernel_size=1),
                                     nn.ConvTranspose2d(in_channels=1,out_channels=1, kernel_size=16, stride=8),
                                     nn.ConvTranspose2d(in_channels=1,out_channels=1, kernel_size=4, stride=2),
                                     nn.ConvTranspose2d(in_channels=1,out_channels=1, kernel_size=4, stride=2))
        
    def forward(self, x):
        x = self.encoder.forward(x)
#         print(x.shape)
        x = self.decoder.forward(x)
#         print(x.shape)
        return x
            #return self.encoder.forward(x)