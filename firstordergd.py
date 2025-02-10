import torch
from PIL import Image
from torchvision import transforms, models

# Load Image and convert it to tensor 3x128x128
SZ = 128
auth = Image.open("mario.png").resize((SZ, SZ)).convert('RGB')
transform = transforms.ToTensor()
transformBack = transforms.ToPILImage()
Y = transform(auth)

# Define similarity functions
def l2_similarity(X, Y):
    return 1 - torch.sqrt(torch.mean((X - Y) ** 2))

def cosine_similarity(X, Y):
    return torch.cosine_similarity(X.view(-1), Y.view(-1), dim=0)

# Define different similarity functions based on difficulty
task = 'hard'  # Change to 'simple', 'advanced', or 'hard'
if task == 'simple':
    similarity = l2_similarity
elif task == 'advanced':
    similarity = cosine_similarity
elif task == 'hard':
    net = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Identity()
    similarity = lambda X, Y: torch.cosine_similarity(net(X.unsqueeze(0))[0], net(Y.unsqueeze(0))[0], dim=0).detach()

# Define the loss function (Pixel-wise MSE for better reconstruction)
def loss_function(X, Y):
    return torch.nn.functional.mse_loss(X, Y)

# Hyperparameters
max_iters = 5000  # Maximum number of iterations
learning_rate = 0.005  # Lower learning rate 
cutoff = 0.001  # MSE threshold for stopping criteria

# Initialize X as a blurred version of the target image instead of random noise
X = Y.clone() + 0.1 * torch.randn_like(Y)  # Add small noise for optimization
X.requires_grad = True
optimizer = torch.optim.Adam([X], lr=learning_rate)

# Gradient Descent Loop
best_loss = loss_function(X, Y).item()
for i in range(max_iters):
    optimizer.zero_grad()
    loss = loss_function(X, Y)  # Minimize pixel-wise difference
    loss.backward()
    optimizer.step()
    
    # Clamp values between [0, 1] to ensure valid image
    with torch.no_grad():
        X.clamp_(0, 1)
    
    current_loss = loss.item()
    if current_loss < best_loss:
        best_loss = current_loss
    
    if best_loss < cutoff:
        break

# Calculate final similarity scores
l2_sim = l2_similarity(X.detach(), Y).item()
cos_sim = cosine_similarity(X.detach(), Y).item()
task_sim = similarity(X.detach(), Y).item()

# Save the final image
final_image = transformBack(X.detach())
final_image.save("final_image.png")
print(f"Succeeded in {i+1} iterations, final MSE loss: {best_loss:.6f}")
print(f"L2 Similarity: {l2_sim:.6f}, Cosine Similarity: {cos_sim:.6f}, Task Similarity ({task}): {task_sim:.6f}")
print("Final image saved as final_image.png")
