# Diskussion
## Grundlage der KI
- Sprachmodell als Basis, ähnlich wie ChatGPT
- Basismodell wird spezialisiert, zugeschnitten auf Textklassifikation: Text als Input, Delikt-Label als Output
- wahrscheinlichkeitsbasiert:
  - für jeden Text wird für jedes mögliche Delikt eine Wahrscheinlichkeit ausgegeben, d.h. 17 verschiedene Werte pro Text
  - am Ende wird das Delikt mit der höchsten Wahrscheinlichkeit ausgegeben
## Trainingsdaten
- 260 manuell annotierte Texte
- aus den Büchern 1-8 der RPG
  - convenience sample: da sie als XML-Daten vorlagen und Bücher 9-11 nur online abrufbar waren)
## Analysierte Delikte
Tötungsdelikte, Körperverletzung, Raubzüge/Kriegsbeteiligung, Klosterangelegenheiten, Lockerung von Fastenvorschriften, Gottesdienst trotz Exkommunikation/Interdikt, Kirchliche Karriere (Promotionsangelegenheiten), Besuchserlaubnis für Pilgerstätten, Simonie (Ämterkauf), Körperlicher Defekt, Absolution von Sentenzen, Lockerung oder Umwandlung von Gelübden, Übriges, Beichtprivileg, Bestattung, Sonderregel Messe, Sex
## Ergebnis
94,6% korrekte Zuordnungen in den Trainingsdaten
## Raum für Verbesserungen
### Maximale Länge des Eingabetextes
- aktuell: 50 Tokens (entspricht etwa 30 Wörtern)
- alles jenseits der ersten 30 Wörter eines Textes wird ignoriert
    > **Vorteil:** hohe Verarbeitungsgeschwindigkeit (ca. 200 Texte pro Sekunde auf einem handelsüblichen Notebook mit guter Hardware-Ausstattung)
    
    > **Nachteil:** längere Texte werden möglicherweise nicht korrekt eingeordnet, weil die entscheidenden Informationen erst später im Text auftauchen
    
    > **Alternative:** anderes KI-Modell als Grundlage für das Training nutzen, mit der Möglichkeit zur Verarbeitung längerer Texte
    > > Auswirkung: Training dauert länger, bessere Hardware nötig
### Evaluation bezogen auf Textlänge
- Werden längere Texte wirklich systematisch schlechter zugeordnet als kürzere? Oder hat dieser Unterschied kaum Auswirkungen auf das Ergebnis?
### Nutzung weiterer Trainingsdaten
- annotierte Beispiele aus der Online-Datenbank abrufen und in das Training miteinbeziehen
### Reduzierung der Kategorien
- weniger Delikt-Labels, die zur Auswahl stehen, bedeuten leichteres/besseres Training für den Algorithmus
- manche Delikte könnten unter einer Obergruppe zusammengefasst werden
  - Gewaltdelikte (Tötungsdelikte, Körperverletzung, Raubzüge)
  - Kirchliche Karriere (Kirchliche Karriere, Simonie, Körperlicher Defekt)
  - Sondererlaubnis (Lockerung von Fastenvorschriften, Besuchserlaubnis, Lockerung/Umwandlung von Gelübden, Beichtprivilegien, Sonderregel Messe)
  > **Problem:** Nivellierung von Nuancen zwischen den verschiedenen Delikttypen, dadurch weniger Aussagekraft der Ergebnisse