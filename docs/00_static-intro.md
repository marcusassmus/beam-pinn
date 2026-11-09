## BEAM PINN STATIC INTRO

Zum Start gibt es hier eine simple Benchmark-Implementierung zur Lösung der Euler-Bernoulli-Balkendifferentialgleichung 4. Ordnung mithilfe von Physics-Informed Neural Networks (PINNs) in PyTorch. 

Dieses Projekt demonstriert, wie neuronale Netze **höhere automatische Differenzierung (bis zur 4. Ableitung)** nutzen können, um physikalische Gesetze (Gleichgewichtsbedingungen und Randbedingungen) direkt in die Verlustfunktion (Loss Function) $\mathcal{L}$ zu integrieren.

**Warum?** Man prüft eine neue Methode zuerst an einem Problem mit bekannter Lösung. Die analytische Lösung geht hier nicht ins Training ein, sondern wird nur zum Vergleich benutzt. Das ist also eine saubere Verifikation, und man sieht dabei, wie schnell die Genauigkeit mit steigender Ableitungsordnung abnimmt. Diese Frage ist für Anwendungen relevant, bei denen Momente und Querkräfte interessieren und nicht nur die Verschiebung.


## Hauptmerkmale

* **Formulierung:** Löst die kontinuierliche Durchbiegung eines Kragbalkens unter einer Gleichlast $q$ rein über physikalisch informierte Verlustoptimierung.
* **Validierung höherer Autograd-Ableitungen:** Nutzt PyTorchs `torch.autograd.grad` rekursiv, um Folgendes zu berechnen:
  -  $0$. Ordnung: Durchbiegung $w(x)$
  -  $1$. Ordnung: Neigung / Verdrehung $\frac{dw}{dx}=\alpha(x)$
  -  $2$. Ordnung: Krümmung $\frac{d^2w}{dx^2}=\kappa(x)$, daraus das Biegemoment $M(x) = -EI\,\frac{d^2w}{dx^2}$
  -  $3$. Ordnung: $\frac{d^3w}{dx^3}$, daraus die Querkraft $Q(x) = -EI\,\frac{d^3w}{dx^3}$
  -  $4$. Ordnung: Lastgleichgewicht $\frac{d^4w}{dx^4} = \frac{q(x)}{EI}$

  (Vorzeichenkonvention: $M' = Q$, $Q' = -q$. Im Skript gilt $EI = 1$; die Plots zeigen daher $w''$ und $w'''$, also $-M$ und $-Q$.)
* **Multi-Plot-Analyse:** Generiert eine strukturierte **Auswertungsmatrix**, die die PINN-Vorhersagen den exakten analytischen Lösungen gegenüberstellt, ergänzt durch individuelle Fehlerprofile und den Konvergenzverlauf des Trainings-Loss.



## Mathematischer Hintergrund

Die zugrundeliegende Differentialgleichung für einen Bernoulli-Balken unter einer Streckenlast $q$ lautet:

$$
EI \frac{d^4 w(x)}{dx^4} = q(x)
$$

**Koordinatensystem und Vorzeichen:** $x$ läuft entlang der Balkenachse von der Einspannung ($x=0$) zum freien Ende ($x=L$); $w$ und $q$ sind in dieselbe Richtung positiv (hier: nach unten). Positiv ist das Biegemoment bei Zug an der unteren Faser. Dann gilt $M = -EI\,w''$, $Q = M' = -EI\,w'''$ und $Q' = -q$. Bei umgekehrter Definition kehren sich die Vorzeichen um.

Mit den Randbedingungen für einen Kragbalken (feste Einspannung bei $x=0$, freies Ende bei $x=L$):
* **Feste Einspannung ($x = 0$):** Durchbiegung $w(0) = 0$, Neigung $\frac{dw}{dx}(0) = 0$
* **Freies Ende ($x = L$):** Biegemoment $M(L) = -EI\,w''(L) = 0$ und Querkraft $Q(L) = -EI\,w'''(L) = 0$, bei konstantem $EI$ also $w''(L) = 0$ und $w'''(L) = 0$



## Verlustfunktion

Die Verlustfunktion setzte sich aus Anteilen der Differentialgleichung, der Randbedingungen und eventueller Daten zusammen.

$$
\mathcal{L}=\mathcal{L}_{\text{PDE}}+\lambda_{\text{BC}} \cdot \mathcal{L}_{\text{BC}}
$$


Die Erfüllung des Lastgleichgewichts wird an $N_{f}$ stochastisch gewürfelten Kollokationspunkten im Inneren des Balkens überprüft:

$$
\mathcal{L}_{\text{PDE}}=\frac{1}{N_{f}}\sum _{i=1}^{N_{f}}\left(EI\frac{\mathrm{d}^{4}w_{\theta }(x_{i})}{\mathrm{d}x^{4}}-q(x_{i})\right)^{2}
$$

Da das Moment $M(L)$ und die Querkraft $Q(L)$ am freien Balkenende verschwinden müssen, ergeben sich die verbleibenden Strafterme (Soft Constraints) aus der 2. und 3. Ableitung. Zur Stabilisierung des Trainings werden diese mit einem Gewichtungsfaktor (z. B. $\lambda_{\text{BC}} = 150$) versehen:

$$
\mathcal{L}_{\text{BC}}=\left(\left.\frac{\mathrm{d}^{2}w_{\theta }}{\mathrm{d}x^{2}}\right|{}_{x=L}\right)^{2}+\left(\left.\frac{\mathrm{d}^{3}w_{\theta }}{\mathrm{d}x^{3}}\right|{}_{x=L}\right)^{2}
$$


## Ablauf

Das physikalisch-informierte Training läuft in der aktuellsten Version über eine optimierte Pipeline aus vier Bausteinen ab:

* **Netzwerk & Hard Constraints:** Ein mehrschichtiges Perzeptron (MLP) mit einer Architektur von 1–40–40–1 Neuronen und glatten `tanh`-Aktivierungsfunktionen approximiert eine Hilfsfunktion $\mathcal{N}(x)$. Um die kinematischen Randbedingungen an der festen Einspannung ($x=0$) mathematisch exakt zu garantieren, wird der Netzwerkausgang direkt im Forward-Pass transformiert:
  
$$
w_{\theta}(x) = x^2 \cdot \mathcal{N}(x)
$$
  
  Dadurch sind $w(0) = 0$ und $w'(0) = 0$ zu jedem Zeitpunkt des Trainings exakt erfüllt, was die Verlustfunktion vereinfacht und konkurrierende Strafterme eliminiert.

* **Robuste Autograd-Ableitungen:** Die Hilfsfunktion `get_derivatives` berechnet die Ableitungen stufenweise über PyTorchs automatische Differenzierung. `torch.autograd.grad` verlangt einen skalaren Ausgang, deshalb wird vor jeder Differenzierungsstufe mit `.sum()` summiert (gleichwertig: `grad_outputs=torch.ones_like(...)`). Das liefert die punktweisen Ableitungen, weil das Netz jeden Punkt $x_i$ unabhängig von den anderen auswertet: Die Jacobi-Matrix $\partial w_i/\partial x_j$ ist diagonal, also gilt $\partial \sum_i w_i / \partial x_j = \partial w_j/\partial x_j$. Das setzt voraus, dass keine Kopplung zwischen den Batch-Punkten besteht (z. B. keine Batch-Normalisierung). Mit `create_graph=True` bleibt der Graph für die nächste Stufe und für die Rückwärtsableitung des Loss erhalten.

* **Zweistufiges Hybrid-Training:** Das Optimierungsverfahren ist in zwei Phasen unterteilt, um globale Stabilität mit lokaler Glattheit zu kombinieren:
  - **Phase 1 (Adam-Exploration):** 4000 Epochen mit dem Adam-Optimierer ($lr = 0.002$). In jeder Epoche werden $N_f = 150$ interne Kollokationspunkte stochastisch neu gewürfelt. Dies zwingt das Netz zur physikalischen Generalisierung und minimiert das PDE-Residuum sowie die weichen Strafterme (Soft Constraints) am freien Balkenende ($x=L$, Gewichtungsfaktor: 150).
  - **Phase 2 (L-BFGS-Glättung):** 25 zusätzliche Schritte mit dem Quasi-Newton-Verfahren L-BFGS (via `closure()`, je Schritt bis zu `max_iter=30` Iterationen) auf einem festen Satz von $N_f = 150$ äquidistanten Kollokationspunkten. L-BFGS nähert die inverse Hesse-Matrix aus den zurückliegenden Gradienten an (Historie: 100) und kann das Minimum auf den festen Punkten deutlich genauer ansteuern als Adam mit wechselnden Punkten. Ziel ist, den Loss auf $< 10^{-6}$ zu senken.

* **Auswertung & Verifikation:** An 100 äquidistanten Testpunkten werden die Durchbiegung und alle vier Ableitungen berechnet. Die Ergebnisse werden mit den geschlossenen analytischen Verläufen des Kragträgers abgeglichen und inklusive ihrer absoluten Fehlerprofile in einer direkten Gegenüberstellung visualisiert.

## Skript
  Das Skript ist wie folgt abgelegt: [00_beam-pinn-static-intro.py](../code/00_beam-pinn-static-intro.py)
  
  Ergebnisdarstellungen sind wie folgt abgelegt: [results/00/](../results/00/)

## Schnellstart
### Google Colab
* Das Skript kann ohne lokale Installation direkt in [Google Colab](https://colab.research.google.com/) ausgeführt werden.
### Lokal
* Um das Skript auf deinem eigenen Rechner auszuführen, wird eine Python-Umgebung (empfohlen: Python $\ge$ 3.8) benötigt. 

1. **Abhängigkeiten installieren:**
   Installiere die erforderlichen mathematischen und Deep-Learning-Bibliotheken über das Terminal/die Eingabeaufforderung:
   ```bash
   pip install torch numpy matplotlib
   ```
2. **Skript ausführen:**
  Das Skript klonen oder in eigene Umgebung kopieren und die Berechnung starten:
   ```bash
   cd code
   python 00_beam-pinn-static-intro.py
   ```
