# Adaptív versenyzés és többautós NEAT-tanítás

Az `adaptive_race.py` külön versenymód: közös pályán, közös időlépésekkel futtatja az autókat. A régi szkriptek működését nem módosítja.

## Telepítés

Python 3.12 ajánlott:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Kipróbálás modell nélkül

```bash
python adaptive_race.py race --cars 4 --collisions --seconds 900 --render
```

A megjelenítéshez grafikus asztal kell; szerveren hagyd el a `--render` kapcsolót. A `--collisions` elhagyásával az autók átjárhatók. Az ellenfélérzékelés mindkét módban aktív. Az eredmények a `runs/adaptive/results.json` fájlba kerülnek: köridők, megtett táv, kontaktusok, pályán kívüli idő és fitness. Az ablak bezárásával a futás leáll, az addigi eredmény elmentődik.

## Tanítás egymás ellen

```bash
python adaptive_race.py train --gens 40 --pop 24 --cars 4 --tracks-per-gen 3 --seconds 900 --collisions --outdir runs/adaptive
python adaptive_race.py race --genome runs/adaptive/adaptive_winner.pkl --cars 4 --seconds 900 --collisions --render --seed 1000001
```

Minden generáció új véletlen pályákat kap. Egy generáción belül ugyanazokon a pályákon mérjük az egyedeket, véletlenül csoportosított rajtrácsokon. A tanítás fitness-e: előrehaladás méterben − 20 × pályán kívüli másodperc − 25 × kontaktusesemény. A hátramenet csökkenti az előrehaladást. A modellek ugyanabban a világban vezetnek, így egy másik egyed döntése megváltoztathatja a helyzetüket. Ez egyszerű populációs együttfejlődés; még nincs külön ellenfélarchívum vagy bajnokság.

Hosszú pályán két kör több perc szimulációt is igényel. A `--seconds` felső időkorlát, nem körszám; a tanításhoz úgy válaszd meg, hogy legyen idő a harmadik körre is. A rövid smoke teszt nem tanít versenyképes modellt.

A `--track track.json` saját zárt pályát fogad: `center`, `left`, `right` Nx2 ponttömbök és `widths` N pozitív szélesség. A régi tréner `replay_track.json` fájlja is használható. Saját pályával a tanítás azon a pályán történik; az általánosítást más pályákon mérd.

## Mit tanul menet közben?

- Az alapvezető a **megadott pályageometria ismeretében** követi a középvonalat, és a kanyarok alapján fékezési profilt készít. Ez nem ismeretlen pálya kizárólag szenzoros feltérképezése.
- Minden autónak külön memóriája van, amely minden új verseny/pálya elején nullázódik. Az első kör a geometriai sebességhatár 65%-ával, a második 82%-ával, a harmadiktól 100%-ával tervez. Pályaelhagyás vagy kontaktus csökkenti az adott szektor eltárolt sebességhatárát.
- A NEAT 36 bemenetet kap: a régi 26 pályaérzékelő, a két legközelebbi ellenfél relatív helyzete/sebessége és jelenléte, a célsebesség és a felmérési fázis. Kormánykorrekciót, célsebesség-módosítást és plusz fékezést tanul. Egy vészfékező követési szabály modell nélkül is működik.
- Futás közben a sebességprofil változik; a hálózat súlyai generációk között változnak. Nincs általános hiperparaméter-keresés, garantált optimum vagy garantált kétkörös megtanulás. A harmadik kör gyorsulása még a beépített sebességlépcsőből is származik, önmagában nem bizonyít tanult általánosítást.
- Az ütközési modell egyszerű, rugalmatlan körtestkontaktus 1,3 m sugárral, pozíciószétválasztással és sebességvesztéssel, sűrű fizikai részlépésekkel. Nem teljes merevtest-fizika. A pályaszélek továbbra is elhagyási büntetést jelentenek, nem fizikai falat.

A korábbi 26 bemenetes modellek az új móddal nem kompatibilisek. Új modell tanítása szükséges. Csak megbízható forrásból származó pickle modellt tölts be. Az adaptív verseny megjelenítője mutatja az összes autót; a régi `racing_live.py` és `replay_neat.py` nem támogatja az új modellt.

## Ellenőrzés

```bash
python -m unittest discover -s tests -v
python adaptive_race.py train --gens 2 --pop 4 --cars 2 --tracks-per-gen 2 --seconds 3 --collisions --outdir /tmp/adaptive-smoke
python adaptive_race.py race --genome /tmp/adaptive-smoke/adaptive_winner.pkl --cars 2 --seconds 3 --collisions --outdir /tmp/adaptive-replay
```

A tesztek ellenőrzik az ütközést, a kapcsolhatóságot, az ellenféladatokat, a memóriák elkülönítését, a körszámlálást/hátramenetet és három kör teljesítését egy körpályán. Ezek funkcionális ellenőrzések; széles körű, ismeretlen pályás teljesítménymérés és hosszabb modelltréning még szükséges.
