"""
Projekt: Physics-Informed Neural Network (PINN) für den Euler-Bernoulli-Balken
Beschreibung: Reines PINN (4. Ordnung) zur Berechnung der Durchbiegung eines 
              Kragbalkens unter Gleichlast mittels Hard Constraints und 
              einem Adam + L-BFGS Hybrid-Optimierer.
Autor: Marcus Aßmus
Datum: April 2026
Lizenz: MIT License
"""



import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Reproduzierbarkeit sicherstellen
torch.manual_seed(42)

# --- Physikalische Parameter ---
L = 1.0       # Balkenlänge [m]
EI = 1.0      # Biegesteifigkeit [Nm^2]
q = 1.0       # Streckenlast [N/m]

# --- 1. Pure PINN Definition mit stabilisierter Aktivierung (Sinus) ---
class PureBeamPINN(nn.Module):
    def __init__(self):
        super().__init__()
        # Wir nutzen torch.sin für glatte höhere Ableitungen
        self.net = nn.Sequential(
            nn.Linear(1, 40),
            nn.Tanh(),  # Tanh kombiniert mit ausreichend Epochen liefert hier das beste Signal
            nn.Linear(40, 40),
            nn.Tanh(),
            nn.Linear(40, 1)
        )
        
    def forward(self, x):
        # Hard Constraint Ansatz: Wir zwingen w(0) = 0 und w'(0) = 0 direkt in die Architektur!
        # Ein Ansatz der Form w(x) = x^2 * N(x) erfüllt die Einspannung bei x=0 immer perfekt.
        return (x ** 2) * self.net(x)

model = PureBeamPINN()

# --- 2. Hilfsfunktion für die reinen Ableitungen ---
def get_derivatives(model, x):
    w = model(x)
    
    # [0] extrahiert den Tensor aus dem von autograd zurückgegebenen Tupel
    dw_dx = torch.autograd.grad(w.sum(), x, create_graph=True)[0]
    d2w_dx2 = torch.autograd.grad(dw_dx.sum(), x, create_graph=True)[0]
    d3w_dx3 = torch.autograd.grad(d2w_dx2.sum(), x, create_graph=True)[0]
    d4w_dx4 = torch.autograd.grad(d3w_dx3.sum(), x, create_graph=True)[0]
    
    return w, dw_dx, d2w_dx2, d3w_dx3, d4w_dx4


# --- 3. Training Loop mit Adam + L-BFGS ---
epochs_adam = 4000
optimizer_adam = torch.optim.Adam(model.parameters(), lr=0.002)
optimizer_lbfgs = torch.optim.LBFGS(model.parameters(), lr=0.05, max_iter=30, history_size=100)

loss_history = []

print("Starte Adam Basis-Training...")
for epoch in range(epochs_adam):
    optimizer_adam.zero_grad()
    
    # Kollokationspunkte im Inneren
    x_interior = torch.rand(150, 1, requires_grad=True) * L
    _, _, _, _, d4w_dx4 = get_derivatives(model, x_interior)
    loss_pde = torch.mean((EI * d4w_dx4 - q) ** 2)
    
    # Randbedingungen am freien Ende (x = L): w''(L) = 0, w'''(L) = 0
    # (x = 0 ist durch die Netzwerk-Architektur oben bereits exakt Null!)
    x_bcL = torch.tensor([[L]], requires_grad=True)
    _, _, d2w_bcL, d3w_bcL, _ = get_derivatives(model, x_bcL)
    loss_bcL = d2w_bcL**2 + d3w_bcL**2
    
    loss = loss_pde + 150.0 * loss_bcL
    loss.backward()
    optimizer_adam.step()
    loss_history.append(loss.item())
    
    if epoch % 1000 == 0:
        print(f"Adam Epoch {epoch}, Loss: {loss.item():.6f}")

print("\nStarte L-BFGS Krümmungs-Glättung...")
def closure():
    optimizer_lbfgs.zero_grad()
    x_int = torch.linspace(0, L, 150).view(-1, 1).requires_grad_(True)
    _, _, _, _, d4w_col = get_derivatives(model, x_int)
    l_pde = torch.mean((EI * d4w_col - q) ** 2)
    
    x_L = torch.tensor([[L]], requires_grad=True)
    _, _, d2w_L, d3w_L, _ = get_derivatives(model, x_L)
    l_bcL = d2w_L**2 + d3w_L**2
    
    total_loss = l_pde + 150.0 * l_bcL
    total_loss.backward()
    return total_loss

for i in range(25):
    loss = optimizer_lbfgs.step(closure)
    loss_history.append(loss.item())
    print(f"L-BFGS Step {i}, Loss: {loss.item():.6f}")

# --- 4. Auswertung  ---
x_test = torch.linspace(0, L, 100).view(-1, 1).requires_grad_(True)
w_t, dw_t, d2w_t, d3w_t, d4w_t = get_derivatives(model, x_test)

x_np = x_test.detach().numpy()
w_pinn = w_t.detach().numpy()
dw_pinn = dw_t.detach().numpy()
d2w_pinn = d2w_t.detach().numpy()
d3w_pinn = d3w_t.detach().numpy()
d4w_pinn = d4w_t.detach().numpy()

# Analytische Lösungen
w_ex = (q / (24 * EI)) * (x_np**4 - 4 * L * x_np**3 + 6 * (L**2) * x_np**2)
dw_ex = (q / (6 * EI)) * (x_np**3 - 3 * L * x_np**2 + 3 * (L**2) * x_np)
d2w_ex = (q / (2 * EI)) * (x_np**2 - 2 * L * x_np + L**2)
d3w_ex = (q / EI) * (x_np - L)
d4w_ex = np.full_like(x_np, q / EI)

# --- 5. Ergebnisdarstellungen ---

#  Plot für den Trainingsverlauf (Loss-Konvergenz) 
plt.figure(figsize=(8, 4))
plt.plot(loss_history, color='tab:blue', linewidth=1.5, label="Gesamt-Loss")
plt.title("Training Loss Konvergenz (Adam + L-BFGS Feinabstimmung)", fontsize=12)
plt.xlabel("Optimierungsschritt (Epoche / Step)", fontsize=10)
plt.ylabel("Loss (logarithmisch)", fontsize=10)
plt.yscale('log')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.tight_layout()
plt.show()


# Matrix-Layout für Ergebnisse von w und Ableitungen
fig, axs = plt.subplots(5, 2, figsize=(14, 18))

def plot_comparison(ax, x, y_ex, y_pinn, title):
    ax.plot(x, y_ex, 'k--', label="Analytisch", linewidth=2, zorder=3)
    ax.plot(x, y_pinn, 'r-', label="Pure PINN (4th Order)", linewidth=2, zorder=2)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

plot_comparison(axs[0, 0], x_np, w_ex, w_pinn, "0. Ableitung: Durchbiegung w(x)")
axs[0, 1].plot(x_np, np.abs(w_pinn - w_ex), 'b-')
axs[0, 1].grid(True)

plot_comparison(axs[1, 0], x_np, dw_ex, dw_pinn, "1. Ableitung: Neigung dw/dx")
axs[1, 1].plot(x_np, np.abs(dw_pinn - dw_ex), 'b-')
axs[1, 1].grid(True)

plot_comparison(axs[2, 0], x_np, d2w_ex, d2w_pinn, "2. Ableitung: Krümmung d²w/dx²")
axs[2, 1].plot(x_np, np.abs(d2w_pinn - d2w_ex), 'b-')
axs[2, 1].grid(True)

plot_comparison(axs[3, 0], x_np, d3w_ex, d3w_pinn, "3. Ableitung: Querkraft d³w/dx³")
axs[3, 1].plot(x_np, np.abs(d3w_pinn - d3w_ex), 'b-')
axs[3, 1].grid(True)

plot_comparison(axs[4, 0], x_np, d4w_ex, d4w_pinn, "4. Ableitung: Lastgleichgewicht d⁴w/dx⁴")
axs[4, 0].set_xlabel("x [m]")
axs[4, 1].plot(x_np, np.abs(d4w_pinn - d4w_ex), 'b-')
axs[4, 1].set_xlabel("x [m]")
axs[4, 1].grid(True)

plt.tight_layout()
plt.show()