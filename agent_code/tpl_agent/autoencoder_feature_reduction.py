import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_layer_sizes, bottleneck_dim):
        """
        Initialize simple MLP based Autoencoder model.
        
        :param input_dim: Dimension of the input features.
        :param hidden_layer_sizes: List of sizes for hidden layers in the encoder and decoder.
        :param bottleneck_dim: Dimension of the bottleneck layer.
        """
        super(Autoencoder, self).__init__()

        # Encoder
        encoder_layers = []
        current_dim = input_dim
        for hidden_size in hidden_layer_sizes:
            encoder_layers.append(nn.Linear(current_dim, hidden_size))
            encoder_layers.append(nn.ReLU())
            current_dim = hidden_size
        
        encoder_layers.append(nn.Linear(current_dim, bottleneck_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        current_dim = bottleneck_dim
        for hidden_size in reversed(hidden_layer_sizes):
            decoder_layers.append(nn.Linear(current_dim, hidden_size))
            decoder_layers.append(nn.ReLU())
            current_dim = hidden_size

        decoder_layers.append(nn.Linear(current_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    

### convolution based AE
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvAutoencoder(nn.Module):
    def __init__(self, input_channels=3, bottleneck_dim=16):
        """
        Initialize the Convolutional Autoencoder model.

        :param input_channels: Number of input channels in the images (3 for RGB, 1 for grayscale).
        :param bottleneck_dim: Dimension of the bottleneck layer.
        """
        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),  # (batch, 32, 16, 16)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (batch, 64, 8, 8)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (batch, 128, 4, 4)
            nn.ReLU(),
            nn.Flatten(),  # Flatten the tensor for the fully connected layer
            nn.Linear(128 * 4 * 4, bottleneck_dim)  # Bottleneck layer with dimension 16
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 128 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (128, 4, 4)),  # Reshape back to the (batch, 128, 4, 4)
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch, 64, 8, 8)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch, 32, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch, 3, 32, 32)
            nn.Tanh()  # To ensure output is in the range [-1, 1]
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Example usage:
# model = ConvAutoencoder(input_channels=3, bottleneck_dim=16)
# output = model(input_tensor)
