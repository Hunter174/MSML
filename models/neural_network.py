import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(4, 16)  # Input layer to hidden layer
        self.fc2 = nn.Linear(16, 3)  # Hidden layer to output layer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    # Evaluate the loss surface over a grid centered around current weights
    def evaluate_loss_surface(self, X, y, center_weights, direction1, direction2, grid_size=50, step_size=0.01):
        losses = np.zeros((grid_size, grid_size))

        for i in range(grid_size):
            for j in range(grid_size):
                alpha = (i - grid_size // 2) * step_size
                beta = (j - grid_size // 2) * step_size

                # Apply perturbations centered around the current weights
                perturbed_params = []
                for param, cw, d1, d2 in zip(self.parameters(), center_weights, direction1, direction2):
                    perturbed_param = cw + alpha * d1 + beta * d2
                    perturbed_params.append(perturbed_param)

                # Set perturbed parameters to the model
                with torch.no_grad():
                    for p, perturbed in zip(self.parameters(), perturbed_params):
                        p.copy_(perturbed)

                # Compute loss for the perturbed parameters
                outputs = self.forward(X)
                loss = self.criterion(outputs, y).item()
                losses[i, j] = loss

        # Restore the original weights after perturbation
        with torch.no_grad():
            for p, cw in zip(self.parameters(), center_weights):
                p.copy_(cw)

        return losses

    # Project current parameters onto the directions for plotting
    def project_params_onto_directions(self, center_weights, direction1, direction2):
        alpha = 0.0
        beta = 0.0

        # Project current parameters onto the chosen directions
        with torch.no_grad():
            for param, cw, d1, d2 in zip(self.parameters(), center_weights, direction1, direction2):
                alpha += torch.sum((param - cw) * d1).item()
                beta += torch.sum((param - cw) * d2).item()

        return alpha, beta

    # Train and visualize loss surface following the weight path
    def train_with_loss_surface(self, X_train, y_train, num_epochs=100, grid_size=50, step_size=0.01):
        losses_over_time = []

        # Initialize plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Random directions for weight perturbation
        direction1 = [torch.randn_like(p) for p in self.parameters()]
        direction2 = [torch.randn_like(p) for p in self.parameters()]

        # To track the weight path
        path_alphas = []
        path_betas = []
        path_losses = []  # Track the z-value (loss) for the weight path

        for epoch in range(num_epochs):
            # Get current weights as center of the surface
            center_weights = [param.clone().detach() for param in self.parameters()]

            # Forward pass and loss computation
            self.optimizer.zero_grad()
            outputs = self.forward(X_train)
            loss = self.criterion(outputs, y_train)

            # Backward pass and update
            loss.backward()
            self.optimizer.step()

            # Record the loss
            losses_over_time.append(loss.item())

            # Project current weights onto the direction space
            alpha, beta = self.project_params_onto_directions(center_weights, direction1, direction2)

            # Store the alpha, beta, and loss coordinates for plotting the path
            path_alphas.append(alpha)
            path_betas.append(beta)
            path_losses.append(loss.item())  # Track the actual loss at this step

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

                # Evaluate the loss surface centered around current weights
                loss_surface = self.evaluate_loss_surface(X_train, y_train, center_weights, direction1, direction2,
                                                          grid_size, step_size)

                # Create meshgrid for X and Y based on the grid size and step size
                X, Y = np.meshgrid(np.linspace(-grid_size // 2 * step_size, grid_size // 2 * step_size, grid_size),
                                   np.linspace(-grid_size // 2 * step_size, grid_size // 2 * step_size, grid_size))

                # Clear the current plot and plot the updated loss surface
                ax.clear()
                # Plot the convergence point as a reference (red dot)
                ax.scatter(alpha, beta, loss.item(), color='r', s=50, label='Current Weights', zorder=2)

                # Plot the path of weights as they move through the surface (with z-axis as loss)
                if len(path_alphas) > 1:
                    ax.plot(path_alphas, path_betas, path_losses, color='b', label='Weight Path', zorder=3)

                ax.plot_surface(X, Y, loss_surface, cmap='viridis', alpha=0.6,
                                zorder=1)  # Make surface slightly transparent


                # Adjust the plot's limits dynamically to follow the weight path
                ax.set_xlim([-grid_size // 2 * step_size, grid_size // 2 * step_size])
                ax.set_ylim([-grid_size // 2 * step_size, grid_size // 2 * step_size])
                ax.set_zlim([0, max(loss_surface.max(), 2 * loss.item())])

                plt.pause(0.01)  # Pause to update the plot in real time

        plt.show()
