# Tetris-AI
Projektni zadatak iz predmeta Osnovi računarske inteligencije, 2019


- Članovi tima:
   Nenad Mišić, SW-31/2016, Grupa 1


- Asistent:
   Aleksandar Lukić

- Problem koji se rešava:
Implementacija agenta koji igra igru tetris i pokušava da maksimizuje svoj skor. Ideja je da se naprave dva agenta čije se ponašanje ocenjuje i upoređuje. Jedan igra prateći Greedy search algoritam, dok će drugi igrati po algoritmu aproksimiranog Q-učenja.
Skor koji agent dobija biće računat po standardnim pravilima tetrisa, o čemu se detaljnije može videti ovde:  [link](https://tetris.wiki/Scoring)
Ideja je da što više redova spoji u jednom potezu više poena dobija.


Broj spojenih redova | Dobijeni skor
----------------------- | ---------------
1 (single) | 40
2 (double) | 100
3 (triple) | 300
4 (tetris) | 1200

Kako i Greedy search algoritam i aproksimirano Q-učenje zahtevaju opis stanja putem feature-a, izdvdojio sam pojedine kao što su:

1. Akumulirana visina po kolonama
2. Broj linija koje će u tom potezu biti popunjene
3. Broj rupa na tabli 
_Rupa predstavlja prazno polje iznad koga se nalazi bar jedno popunjeno polje_
4. Razuđenost
_Razuđenost predstavlja akumulirane razlike u visinama svake dve susedne kolone_
5. Visina najviše kolone na tabli

Korišćena tehnologija: Python 3 uz pygame biblioteku za iscrtavanje interfejsa
Pomoć pri dizajniranju grafičkog interfejsa je iskorišćena sa sledećeg projekta: [link](https://inventwithpython.com/pygame/chapter7.html)

- Algoritam/algoritmi:
Greedy search algoritam i aproksimirano Q-učenje.
Razlog odabira greedy search-a umesto A* algoritma je to što je cena svakog koraka u igri jednaka, pa nema smisla uzimati je u obzir pri računanju.

- Metrika za merenje performansi:
Poređenje ova dva algoritma međusobno.
Poređenje se izvršava sa jednakim rasporedom padanja figurica, tako da je proces merenja determinističan. Metrika koja će se ocenjivati je skor koji agenti postižu nakon 20, 50, 100 i 500 figurica.

- Validacija rešenja:
Demonstriranje rešenja na odbrani projekta uživo kao i video snimak koji pokazuje scenario igranja na dužem periodu.
