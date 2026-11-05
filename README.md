# Beam PINN
-- physikalisch-informierte neuronale Netze am Bernoulli Biege-Balken --

Zum Start gibt es hier eine simple Benchmark-Implementierung zur Lösung der Euler-Bernoulli-Balkendifferentialgleichung 4. Ordnung mithilfe von Physics-Informed Neural Networks (PINNs) in PyTorch. 

Dieses Projekt demonstriert, wie neuronale Netze **höhere automatische Differenzierung (bis zur 4. Ableitung)** nutzen können, um physikalische Gesetze (Gleichgewichtsbedingungen und Randbedingungen) direkt in die Verlustfunktion (Loss Function) 

$$
\mathcal{L}=\mathcal{L}_{\text{PDE}}+\mathcal{L}_{\text{BC}}+\mathcal{L}_{\text{Data}}
$$

zu integrieren.

**Warum?** Man prüft eine neue Methode zuerst an einem Problem mit bekannter Lösung. Die analytische Lösung geht hier nicht ins Training ein, sondern wird nur zum Vergleich benutzt. Das ist also eine saubere Verifikation, und man sieht dabei, wie schnell die Genauigkeit mit steigender Ableitungsordnung abnimmt. Diese Frage ist für Anwendungen relevant, bei denen Momente und Querkräfte interessieren und nicht nur die Verschiebung.



## Hauptmerkmale

* **Formulierung:** Löst die kontinuierliche Durchbiegung eines Kragbalkens unter einer Gleichlast $q$ rein über physikalisch informierte Verlustoptimierung.
* **Validierung höherer Autograd-Ableitungen:** Nutzt PyTorchs `torch.autograd.grad` rekursiv, um Folgendes zu berechnen:
  -  0. Ordnung: Durchbiegung $w(x)$
  -  1. Ordnung: Neigung / Verdrehung $\frac{dw}{dx}=\alpha(x)$
  -  2. Ordnung: Krümmung / Biegemoment $\frac{d^2w}{dx^2} =M(x)$
  -  3. Ordnung: Querkraft $\frac{d^3w}{dx^3} =Q(x)$
  -  4. Ordnung: Lastgleichgewicht $\frac{d^4w}{dx^4} = \frac{q(x)}{EI}$
* **Multi-Plot-Analyse:** Generiert eine strukturierte **Auswertungsmatrix**, die die PINN-Vorhersagen den exakten analytischen Lösungen gegenüberstellt, ergänzt durch individuelle Fehlerprofile und den Konvergenzverlauf des Trainings-Loss.



## Mathematischer Hintergrund

Die zugrundeliegende Differentialgleichung für einen Bernoulli-Balken unter einer Streckenlast $q$ lautet:

$$
EI \frac{d^4 w(x)}{dx^4} = q(x)
$$

Mit den Randbedingungen für einen Kragbalken (feste Einspannung bei $x=0$, freies Ende bei $x=L$):
* **Feste Einspannung ($x = 0$):** Durchbiegung $w(0) = 0$, Neigung $\frac{dw}{dx}(0) = 0$
* **Freies Ende ($x = L$):** Biegemoment $\frac{d^2w}{dx^2}(L) = 0$, Querkraft $\frac{d^3w}{dx^3}(L) = 0$

Die Komponente der randbedingungen des Verlustes ergeben sich damit folgt:

$$
\mathcal{L}_{\text{BC}}=w_{\theta }(0)^{2}+(w_{\theta }^{\prime }(0))^{2}+(w_{\theta }^{\prime \prime }(L))^{2}+(w_{\theta }^{\prime \prime \prime }(L))^{2}
$$



## Ablauf

* **Netz**: Ein MLP (Mehrschichtiges Perzeptron) mit Architektur 1–32–32–1 (Neuronenanzahl pro Schicht, von Eingang bis Ausgang) und tanh-Aktivierung approximiert $w(x)$. Tanh ist hier sinnvoll, weil die Ableitungen bis zur 4. Ordnung stetig und nicht null sein müssen.
* **Ableitungen**: get_derivatives leitet $w$ per Autograd viermal nach $x$ ab (create_graph=True, damit durch die Ableitungen hindurch rückpropagiert werden kann).
* **Training**: 4000 Epochen Adam ($lr = 0,005$). Der Loss besteht aus
dem PDE-Residuum ($EI·w⁗ − q)^2$, gemittelt über 100 Kollokationspunkte, die jede Epoche neu zufällig gezogen werden,
den vier Randbedingungen als Strafterme (soft constraints), mit Faktor 200 gewichtet.

$$
\mathcal{L}_{\text{PDE}}=\frac{1}{N_{f}}\sum _{i=1}^{N_{f}}\left(EI\frac{{d}^{4}w_{\theta }(x_{i})}{{d}x^{4}}-q(x_{i})\right)^{2}
$$

* **Auswertung**: An 100 äquidistanten Testpunkten werden $w$ und die Ableitungen 1 bis 4 berechnet und mit den analytischen Ausdrücken verglichen. Die Lösung $w = q/(24EI)·(x⁴ − 4Lx³ + 6L²x²)$ und ihre Ableitungen stimmen mit der Standardlösung des Kragträgers überein.



## Schnellstart
### Google Colab
* Du kannst das Skript ohne lokale Installation direkt in [Google Colab](https://colab.research.google.com/) ausführen.
### Lokal
* Das Skript klonen oder in eigene Umgebung kopieren.
* Sicherstellen, dass die erforderlichen Abhängigkeiten installiert sind:
   ```bash
   pip install torch numpy matplotlib
