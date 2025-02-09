import torch
from PIL import Image
from torchvision import transforms, models

# Load Image and convert it to tensor 3x128x128
SZ = 128
auth = Image.open("mario.png").resize((SZ, SZ)).convert('RGB')
transform = transforms.ToTensor()
transformBack = transforms.ToPILImage()
Y = transform(auth)

# Define the similarity function, different versions according to difficulty
task = 'advanced'  # Change to 'simple' or 'advanced' as needed
if task == 'simple':
    similarity = lambda X, Y: 1 - torch.sqrt(torch.mean((X - Y) ** 2))
elif task == 'advanced':
    similarity = lambda X, Y: torch.cosine_similarity(X.view(-1), Y.view(-1), dim=0)
elif task == 'hard':
    net = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Identity()
    similarity = lambda X, Y: torch.cosine_similarity(net(X.unsqueeze(0))[0], net(Y.unsqueeze(0))[0], dim=0)

# Define the score function
def score(X):
    return similarity(X, Y)

# Hyperparameters
max_iters = 5000  # Maximum number of iterations
learning_rate = 0.05  # Step size for gradient descent
cutoff = 0.95  # Target similarity score

# Initialize X as a trainable tensor from the original image (Y)
X = Y.clone().detach().requires_grad_(True)
optimizer = torch.optim.Adam([X], lr=learning_rate)

# Gradient Descent Loop
best_score = score(X).item()
for i in range(max_iters):
    optimizer.zero_grad()
    loss = -score(X)  # We maximize similarity, so minimize negative similarity
    loss.backward()
    optimizer.step()
    
    # Clamp values between [0, 1] to ensure valid image
    with torch.no_grad():
        X.clamp_(0, 1)
    
    current_score = -loss.item()
    
    if current_score > best_score:
        best_score = current_score
    
    if best_score > cutoff:
        break

# Save the final image
final_image = transformBack(X.detach())
final_image.save("final_image.png")
print(f"Succeeded in {i+1} iterations, final similarity score: {best_score:.4f}")
print("Final image saved as final_image.png")