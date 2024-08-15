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

class ConvAutoencoder(nn.Module):
    def __init__(self, input_channels=6, bottleneck_dim=16, image_size=5, num_channels_hidden_layer=[32, 64]):
        """
        Initialize the Convolutional Autoencoder model.

        :param input_channels: Number of input channels in the images (6 in your case).
        :param bottleneck_dim: Dimension of the bottleneck layer.
        :param image_size: Size of the height and width of the input images (height and width are the same).
        :param num_channels_hidden_layer: List of integers representing the number of channels in hidden layers.
        """
        super(ConvAutoencoder, self).__init__()

        self.image_size = image_size
        self.num_channels_hidden_layer = num_channels_hidden_layer
        self.num_nodes_linear_layer = num_channels_hidden_layer[1]*image_size*image_size

        # Define the Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, num_channels_hidden_layer[0], kernel_size=3, stride=1, padding=1),  # (batch, num_channels_hidden_layer[0], image_size, image_size)
            nn.ReLU(),
            nn.Conv2d(num_channels_hidden_layer[0], num_channels_hidden_layer[1], kernel_size=3, stride=1, padding=1),  # (batch, num_channels_hidden_layer[1], image_size, image_size)
            nn.ReLU(),
            nn.Flatten(),  # Flatten for fully connected layer
            nn.Linear(self.num_nodes_linear_layer, bottleneck_dim)  # Bottleneck dimension
        )

        # Define the Decoder
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, num_channels_hidden_layer[1] * image_size * image_size),
            nn.ReLU(),
            nn.Unflatten(1, (num_channels_hidden_layer[1], image_size, image_size)),  # Reshape back to feature map dimensions
            nn.ConvTranspose2d(num_channels_hidden_layer[1], num_channels_hidden_layer[0], kernel_size=3, stride=1, padding=1),  # (batch, num_channels_hidden_layer[0], image_size, image_size)
            nn.ReLU(),
            nn.ConvTranspose2d(num_channels_hidden_layer[0], input_channels, kernel_size=3, stride=1, padding=1),  # (batch, input_channels, image_size, image_size)
            nn.Tanh()  # Output range [-1, 1]
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded