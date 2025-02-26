## TENSORFLOW VERSION
# import tensorflow as tf
# from tensorflow.keras import Input
# from tensorflow.keras.models import Model, load_model, save_model
# from tensorflow.keras.layers import (
#     Input,
#     Activation,
#     BatchNormalization,
#     Dropout,
#     Lambda,
#     Conv2D,
#     Conv2DTranspose,
#     MaxPooling2D,
#     concatenate,
# )
# from tensorflow.keras import backend as K

# def unet(input_size=(256, 256, 3)):
#     inputs = Input(input_size)

#     # Encoding Leg
#     '''Filters: The filters=64 parameter tells the layer to learn 64 different filters. Each filter will detect different features (like edges or textures) in the input.
#         Kernel Size: The kernel_size=(3,3) indicates that each filter is a 3x3 matrix. This small window slides over the entire input image or feature map to perform localized processing.
#         Padding: The padding='same' parameter means that the output will have the same spatial dimensions (height and width) as the input. This is achieved by adding zeros around the border of the input when needed.
#         Operation: The layer applies these 64 filters across the input. At every position, the filter performs an element-wise multiplication with the part of the input it covers, sums the results, and produces a single value. This process creates a new feature map that highlights certain features learned by the filter.'''
#     conv1 = Conv2D(filters=64, kernel_size=(3,3), padding='same')(inputs)
#     # batch normalization
#     bn1 = Activation('relu')(conv1) #setting all negative values to zero, ReLU helps prevent issues like vanishing gradients and speeds up the training process.
#     conv1 = Conv2D(filters=64, kernel_size=(3,3), padding='same')(bn1)
#     bn1 = Activation('relu')(conv1)
#     bn1 = Activation('relu')(bn1)
#     #MaxPooling2D(pool_size=(2,2)) takes non-overlapping 2×2 blocks and keeps only the maximum value from each block. As a result, it halves both the height and width of the feature maps.
#     pool1 = MaxPooling2D(pool_size=(2,2))(bn1)

#     conv2 = Conv2D(filters=128, kernel_size=(3,3), padding='same')(pool1)
#     # batch normalization
#     bn2 = Activation('relu')(conv2)
#     conv2 = Conv2D(filters=128, kernel_size=(3,3), padding='same')(bn2)
#     bn2 = Activation('relu')(conv2)
#     bn2 = Activation('relu')(bn2)
#     pool2 = MaxPooling2D(pool_size=(2,2))(bn2)

#     conv3 = Conv2D(filters=256, kernel_size=(3,3), padding='same')(pool2)
#     # batch normalization
#     bn3 = Activation('relu')(conv3)
#     conv3 = Conv2D(filters=256, kernel_size=(3,3), padding='same')(bn3)
#     bn3 = Activation('relu')(conv3)
#     bn3 = Activation('relu')(bn3)
#     pool3 = MaxPooling2D(pool_size=(2,2))(bn3)

#     conv4 = Conv2D(filters=512, kernel_size=(3,3), padding='same')(pool3)
#     # batch normalization
#     bn4 = Activation('relu')(conv4)
#     conv4 = Conv2D(filters=512, kernel_size=(3,3), padding='same')(bn4)
#     bn4 = BatchNormalization(axis=3)(conv4)
#     bn4 = Activation('relu')(bn4)
#     pool4 = MaxPooling2D(pool_size=(2,2))(bn4)

#     conv5 = Conv2D(filters=1024, kernel_size=(3,3), padding='same')(pool4)
#     bn5 = Activation('relu')(conv5)
#     conv5 = Conv2D(filters=1024, kernel_size=(3,3), padding='same')(bn5)
#     bn5 = BatchNormalization(axis=3)(conv5)
#     bn5 = Activation('relu')(bn5)

#     # Decoding Leg with Conv Transpose
#     '''ow UpConvolution / Decoder Leg will begin, so start with Conv2DTranspose
#     The gray arrows (in the above image) indicate the skip connections that concatenate the encoder feature map with the decoder, which helps the backward flow of gradients for improved training.'''
#     '''strides=(2,2) with Conv2DTranspose doubles the spatial dimensions (height/width) of the input feature map, helping the network reconstruct higher-resolution features.'''
#     # In Keras, the input tensors typicaly have the shape (batch_size, height, width, channels).
#     # The axis=3 parameter specifies that the concatenation should be done along the channel dimension.
#     conv6 = concatenate([Conv2DTranspose(512, kernel_size=(2,2), strides=(2,2), padding='same')(bn5), conv4], axis=3)
#     bn6 = Activation('relu')(conv6)
#     conv6 = Conv2D(filters=512, kernel_size=(3,3), padding='same')(bn6)
#     # batch normalization is used to normalize the input data to have near zero mean and unit variance, normalzie for each mini batch
#     bn6 = BatchNormalization(axis=3)(conv6)
#     bn6 = Activation('relu')(bn6)

#     up7 = concatenate(
#         [
#             Conv2DTranspose(256, kernel_size=(2, 2), strides=(2, 2), padding="same")(
#                 bn6
#             ),
#             conv3,
#         ],
#         axis=3,
#     )
#     conv7 = Conv2D(filters=256, kernel_size=(3, 3), padding="same")(up7)
#     bn7 = Activation("relu")(conv7)
#     conv7 = Conv2D(filters=256, kernel_size=(3, 3), padding="same")(bn7)
#     bn7 = BatchNormalization(axis=3)(conv7)
#     bn7 = Activation("relu")(bn7)

#     up8 = concatenate(
#         [
#             Conv2DTranspose(128, kernel_size=(2, 2), strides=(2, 2), padding="same")(
#                 bn7
#             ),
#             conv2,
#         ],
#         axis=3,
#     )
#     conv8 = Conv2D(filters=128, kernel_size=(3, 3), padding="same")(up8)
#     bn8 = Activation("relu")(conv8)
#     conv8 = Conv2D(filters=128, kernel_size=(3, 3), padding="same")(bn8)
#     bn8 = BatchNormalization(axis=3)(conv8)
#     bn8 = Activation("relu")(bn8)

#     up9 = concatenate(
#         [
#             Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2), padding="same")(
#                 bn8
#             ),
#             conv1,
#         ],
#         axis=3,
#     )
#     conv9 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(up9)
#     bn9 = Activation("relu")(conv9)
#     conv9 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(bn9)
#     bn9 = BatchNormalization(axis=3)(conv9)
#     bn9 = Activation("relu")(bn9)

#     # Last Layer
#     conv10 = Conv2D(filters=1, kernel_size=(1,1), activation='sigmoid')(bn9)

#     return Model(inputs=[inputs], outputs=[conv10])

## PYTORCH VERSION + MPS
import torch
import torch.nn as nn
import torch.nn.functional as F

# Choose MPS if available
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def conv_block(in_channels, out_channels, use_batchnorm=True):
    """
    A block of:
        Conv2d -> ReLU -> Conv2d -> (BatchNorm) -> ReLU
    """
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
    ]
    if use_batchnorm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        """
        4-level U-Net matching your TensorFlow code's structure:
        - Encoder with 64, 128, 256, 512 filters
        - Bottleneck with 1024 filters
        - Decoder with transposed convolutions and skip connections
        - Final 1×1 conv -> Sigmoid
        """
        super(UNet, self).__init__()

        # ---------------------
        #    Encoder (Down)
        # ---------------------
        self.down1 = conv_block(in_channels, 64, use_batchnorm=False)  # matches no BN in the first conv block
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down2 = conv_block(64, 128, use_batchnorm=False)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down3 = conv_block(128, 256, use_batchnorm=False)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down4 = conv_block(256, 512, use_batchnorm=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck (1024 filters)
        self.bottleneck = conv_block(512, 1024, use_batchnorm=True)

        # ---------------------
        #    Decoder (Up)
        # ---------------------
        # Each up block: TransposeConv -> concat skip -> conv_block
        self.up6_trans = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up6_conv = conv_block(1024, 512, use_batchnorm=True)

        self.up7_trans = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up7_conv = conv_block(512, 256, use_batchnorm=True)

        self.up8_trans = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up8_conv = conv_block(256, 128, use_batchnorm=True)

        self.up9_trans = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up9_conv = conv_block(128, 64, use_batchnorm=True)

        # Final 1x1 convolution
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
        # We apply Sigmoid in forward() for binary segmentation

    def forward(self, x):
        # ---------- Encoder ----------
        c1 = self.down1(x)    # shape: [N, 64, H, W]
        p1 = self.pool1(c1)   # [N, 64, H/2, W/2]

        c2 = self.down2(p1)   # [N, 128, H/2, W/2]
        p2 = self.pool2(c2)   # [N, 128, H/4, W/4]

        c3 = self.down3(p2)   # [N, 256, H/4, W/4]
        p3 = self.pool3(c3)   # [N, 256, H/8, W/8]

        c4 = self.down4(p3)   # [N, 512, H/8, W/8]
        p4 = self.pool4(c4)   # [N, 512, H/16, W/16]

        # ---------- Bottleneck ----------
        c5 = self.bottleneck(p4)  # [N, 1024, H/16, W/16]

        # ---------- Decoder ----------
        # up6
        u6 = self.up6_trans(c5)              # [N, 512, H/8, W/8]
        cat6 = torch.cat([u6, c4], dim=1)    # concat along channel dimension
        c6 = self.up6_conv(cat6)             # [N, 512, H/8, W/8]

        # up7
        u7 = self.up7_trans(c6)              # [N, 256, H/4, W/4]
        cat7 = torch.cat([u7, c3], dim=1)    # [N, 512, H/4, W/4]
        c7 = self.up7_conv(cat7)             # [N, 256, H/4, W/4]

        # up8
        u8 = self.up8_trans(c7)              # [N, 128, H/2, W/2]
        cat8 = torch.cat([u8, c2], dim=1)    # [N, 256, H/2, W/2]
        c8 = self.up8_conv(cat8)             # [N, 128, H/2, W/2]

        # up9
        u9 = self.up9_trans(c8)              # [N, 64, H, W]
        cat9 = torch.cat([u9, c1], dim=1)    # [N, 128, H, W]
        c9 = self.up9_conv(cat9)             # [N, 64, H, W]

        # ---------- Output ----------
        logits = self.final_conv(c9)         # [N, 1, H, W]
        mask = torch.sigmoid(logits)         # for binary segmentation
        return mask
    