# Beam PINN
physikalisch-informierte neuronale Netze am Bernoulli Biege-Balken

Eine simple Benchmark-Implementierung zur Lösung der 4.-Ordnung Euler-Bernoulli-Balkendifferentialgleichung mithilfe von Physics-Informed Neural Networks (PINNs) in PyTorch. 

Dieses Projekt demonstriert, wie neuronale Netze **höhere automatische Differenzierung (bis zur 4. Ableitung)** nutzen können, um physikalische Gesetze (Gleichgewichtsbedingungen und Randbedingungen) direkt in die Verlustfunktion (Loss Function) zu integrieren.

---

## Hauptmerkmale

* **Formulierung:** Löst die kontinuierliche Durchbiegung eines Kragbalkens unter einer Gleichlast $q$ rein über physikalisch informierte Verlustoptimierung.
* **Validierung höherer Autograd-Ableitungen:** Nutzt PyTorchs `torch.autograd.grad` rekursiv, um Folgendes zu berechnen:
  * 0. Ordnung: Durchbiegung $w(x)$
  * 1. Ordnung: Neigung / Verdrehung $\frac{dw}{dx}$
  * 2. Ordnung: Krümmung / Biegemoment $M(x)$
  * 3. Ordnung: Querkraft $Q(x)$
  * 4. Ordnung: Lastgleichgewicht $\frac{d^4w}{dx^4} = \frac{q}{EI}$
* **Multi-Plot-Analyse:** Generiert eine strukturierte **Auswertungsmatrix**, die die PINN-Vorhersagen den exakten analytischen Lösungen gegenüberstellt, ergänzt durch individuelle Fehlerprofile und den Konvergenzverlauf des Trainings-Loss.
* **Google Colab-bereit:** Entwickelt für die schnelle Ausführung und Experimente in cloudbasierten Jupyter-Umgebungen.

---

## Mathematischer Hintergrund

Die zugrundeliegende Differentialgleichung für einen Bernoulli-Balken unter einer Streckenlast $q$ lautet:

$$
EI \frac{d^4 w(x)}{dx^4} = q
$$

Mit den Randbedingungen für einen Kragbalken (feste Einspannung bei $x=0$, freies Ende bei $x=L$):
* **Feste Einspannung ($x = 0$):** Durchbiegung $w(0) = 0$, Neigung $\frac{dw}{dx}(0) = 0$
* **Freies Ende ($x = L$):** Biegemoment $\frac{d^2w}{dx^2}(L) = 0$, Querkraft $\frac{d^3w}{dx^3}(L) = 0$

---

## Schnellstart
### Google Colab
* Du kannst das Skript ohne lokale Installation direkt in [Google Colab](https://colab.research.google.com/) ausführen.
### Lokal
* Das Skript klonen oder in eigene Umgebung kopieren.
* Sicherstellen, dass die erforderlichen Abhängigkeiten installiert sind:
   ```bash
   pip install torch numpy matplotlib
