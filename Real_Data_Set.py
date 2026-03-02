import os
import sys
import re
import time
import wave
import glob
import random
import secrets
import itertools
import tkinter as tk
from tkinter import messagebox
from pathlib import Path
import numpy as np
import sounddevice as sd

if getattr(sys, 'frozen', False):
    BASE_DIR = Path(sys.executable).resolve().parent
else:
    BASE_DIR = Path(__file__).resolve().parent

SEED = secrets.randbits(32)
random.seed(SEED)

OUT_ROOT = BASE_DIR / f"Human_DataSet_{SEED}"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"
VOICE_TAG = "UserMic"

BLOCKSIZE = 1024
POSTROLL_SEC = 0.35

MEDICINES = [
    "Advagraf", "Bisocard", "Dip Rilif", "Dolgit", "Ibuprom Max Sprint",
    "Nurofen Express Forte", "Actisoftin", "Actifed", "Ajovy", "Aimovig",
    "Agrocia", "Agolek", "Afobam", "Afstyla", "Adipine", "Acodin",
    "Acti Vita-miner Prenatal + DHA", "Actilyse 10", "Actilyse 20",
    "Actisorb Plus 25", "ACTI-trin", "Actrapid Penfill", "Adacel Polio",
    "Adadox", "Adadut", "Adaptogeny", "Adatam XR", "Addiphos", "Adenuric",
    "A-Derma Cytelium"
]

LongSentenceValidation_v2 = [
    "W raporcie odnotowano konieczność podania {lek} podopiecznemu",
    "Personel dyżurny proszony jest o włączenie {lek} do stałej karty zleceń, aplikacja po toalecie porannej. {lek} stosujemy interwencyjnie. {lek} zostaje wykreślony z listy, decyzja o przywróceniu zapadnie po wizycie. Należy obserwować czy po {lek} nie występuje nadmierna senność w ciągu dnia.",
    "Utrzymujemy podawanie {lek} przez najbliższe dwa tygodnie, stały rytm ułatwia opiekę nad mieszkańcem. W razie gorszego dnia i płaczliwości można doraźnie zastosować {lek}, notując to w zeszycie raportów. {lek} na ten moment wycofujemy z uwagi na problemy z połykaniem. Gdyby na skórze pojawiła się wysypka po {lek} zaprzestać podawania.",
    "Wprowadzamy {lek} od dzisiejszego dyżuru, zaczynając od połowy tabletki, po tygodniu ocenimy zachowanie seniora. {lek} tylko w sytuacjach skrajnego niepokoju. {lek} omijamy do czasu wykonania badań kontrolnych krwi. Jeśli po {lek} podopieczny będzie miał zawroty głowy, proszę asekurować go przy chodzeniu.",
    "Kontynuujemy {lek} zgodnie z kartą, to obecnie główny preparat uspokajający. {lek} jest opcją awaryjną, proszę wpisywać godzinę podania. {lek} zawieszamy, możliwy powrót w mniejszej ilości po ustabilizowaniu ciśnienia. W razie wymiotów po {lek} proszę podawać go na pełny żołądek i dopilnować wypicia płynu.",
    "Zmiana w karcie, po przebudzeniu {lek} w niskiej ilości, na koniec dnia tylko obserwacja, a {lek} wyłącznie przy agresji słownej. {lek} zostaje w szafce pancernej do przyszłego tygodnia, bo wcześniej obserwowano kołatania serca u seniora. Jeśli po {lek} wystąpi ból brzucha, proszę zrezygnować z podawania.",
    "Wznawiamy {lek} w leczeniu przewlekłym, proszę pilnować regularności. {lek} można podać przed rehabilitacją, jeśli senior zgłasza ból przy ćwiczeniach. {lek} pozostaje w rezerwie, rozważymy powrót tylko przy braku reakcji na {lek}. Proszę nie podawać własnych ziół rodziny, żeby nie zaburzać działania {lek}.",
    "Ustalamy stały czas aplikacji, {lek} po kolacji, a {lek} trzymamy w dyżurce na sytuacje nagłe. {lek} na ten moment jest zakazany, wcześniej powodował upadki w ciągu dnia. Proszę prowadzić kartę obserwacji, zapisywać kiedy senior dostaje {lek} i jak reaguje, to ułatwi korektę leczenia.",
    "Docelowo zwiększymy {lek} o jedną jednostkę po tygodniu, jeśli nie wystąpią reakcje niepożądane. {lek} proszę stosować rozważnie, najlepiej nie później niż o godzinie osiemnastej, aby nie zaburzać rytmu snu pensjonariusza. {lek} wstrzymujemy do czasu konsultacji psychiatrycznej. Jeżeli po {lek} wystąpi duszność, wzywamy pogotowie.",
    "Na dziś zostajemy przy {lek}, bo widać poprawę w zachowaniu podopiecznego. {lek} do wykorzystania przed wizytą rodziny, aby opanować emocje. {lek} wycofujemy, wcześniej powodował omdlenia. Proszę pamiętać o pojeniu seniora, bo {lek} może wysuszać śluzówki.",
    "Proszę rozpocząć podawanie {lek} jutro przy śniadaniu, pierwsze trzy dni w ilości startowej. {lek} tylko gdy ból kręgosłupa uniemożliwia wstanie z łóżka, nie częściej niż co osiem godzin. {lek} będzie rozważony ponownie po wynikach moczu. W dniu podania {lek} proszę wzmóc nadzór nad mieszkańcem, jeśli będzie senny.",
    "Zaczynamy od {lek} w małej ilości na początku zmiany dziennej, proszę podawać z jogurtem. {lek} zostawiam jako ratunkowy w dniach silniejszego pobudzenia. {lek} wstrzymujemy do czasu kontroli, wrócimy do niego tylko jeśli {lek} okaże się za słaby.",
    "Na czas planowanego zabiegu w szpitalu proszę nie podawać {lek} na czterdzieści osiem godzin przed transportem. Utrzymujemy {lek} bez zmian, a {lek} można podać doraźnie, ale nie później niż przed snem. Po powrocie ze szpitala wznowimy {lek}, decyzję o {lek} podejmie lekarz prowadzący.",
    "Senior jest osłabiony, dlatego utrzymamy {lek} w najniższej skutecznej ilości i będziemy monitorować stan. {lek} wyłącznie doraźnie i tylko gdy podopieczny głośno sygnalizuje ból. {lek} na ten moment odkładamy do szafki depozytowej.",
    "Przy karmieniu sondą {lek} należy dokładnie rozkruszyć i podać z wodą. {lek} można zastosować wyjątkowo przy gorączce. {lek} jest wstrzymany, bo wcześniej powodował zaleganie treści żołądkowej.",
    "Z uwagi na stan wątroby mieszkańca zaczynamy {lek} od połówki i obserwujemy kolor skóry i oczu. {lek} doraźnie, ale nie częściej niż dwa razy na dobę. {lek} wycofujemy, wyniki badań są niepokojące.",
    "Przy niewydolności nerek modyfikujemy schemat, {lek} co drugi dzień przez pierwszy tydzień. {lek} tylko w razie potrzeby, proszę zapisywać w raporcie. {lek} na razie niezalecany, czekamy na decyzję nefrologa.",
    "W ramach przygotowania do wyjazdu do rodziny proszę spakować {lek} w kasetkę. {lek} może być użyty przed podróżą, jeśli senior się denerwuje. {lek} pozostaje wycofany, rodzina poinformowana.",
    "Wizyta kontrolna za tydzień, do tego czasu codziennie {lek} o godzinie dwudziestej. {lek} tylko gdy agresja wzrasta. {lek} w rezerwie, rozważymy powrót przy braku efektów {lek}.",
    "Wprowadzamy powolną modyfikację, {lek} zwiększamy po trzech dniach, jeśli senior dobrze toleruje zmianę. {lek} proszę ograniczyć do sytuacji wyjątkowych. {lek} na razie nie, po miesiącu ocenimy bilans.",
    "Jeśli podopieczny śpi po {lek} cały dzień, proszę przenieść podawanie na noc. {lek} może nasilać pragnienie, więc dzbanek z wodą musi być pełny. {lek} był źle tolerowany, dlatego wykreślamy go z karty.",
    "U tego mieszkańca zaczynamy ostrożnie, {lek} w ilości startowej przez tydzień z pomiarem ciśnienia. {lek} na nagłe pogorszenie, ale maksymalnie dwa razy na dobę. {lek} pomijamy, senior się po nim przewracał.",
    "W dni kiedy jest rehabilitant proszę nie zmieniać {lek}, utrzymujemy stałą ilość. {lek} można podać godzinę przed ćwiczeniami. {lek} pozostaje wstrzymany, ocenimy sens powrotu po miesiącu.",
    "Dla bezpieczeństwa proszę nie podawać {lek} razem z witaminami od rodziny. {lek} powodował wymioty, więc rezygnujemy. Gdyby pojawiła się czerwonka plama na skórze po {lek}, należy przerwać podawanie.",
    "Zmieniamy porę, {lek} po śniadaniu, to powinno ożywić seniora w ciągu dnia. {lek} zostaje do użycia awaryjnego, proszę nie podawać po kolacji. {lek} w rezerwie, rozważymy powrót, jeśli bóle nie ustąpią.",
    "Przed szczepieniem na grypę nie trzeba rezygnować z {lek}, podajemy jak zwykle. {lek} można zastosować dzień po szczepieniu w razie gorączki. {lek} nadal wstrzymany, poprzednio nasilał biegunkę.",
    "Jeżeli senior budzi się w nocy, warto przesunąć {lek} o godzinę później. {lek} można dodać raz, jeśli nocne krzyki są uciążliwe. {lek} nie wraca na ten moment, skupiamy się na {lek}.",
    "Po antybiotyku wracamy do stałego podawania {lek}, koniecznie z dużą ilością picia. {lek} wyłącznie interwencyjnie. {lek} wstrzymujemy na kolejny tydzień, by nie obciążać żołądka.",
    "Z uwagi na serce proszę mierzyć puls przed podaniem {lek}. {lek} może podnieść ciśnienie, dlatego unikać podawania na noc. {lek} wycofany, czekamy na e ka gie.",
    "Do czasu wyników badań utrzymujemy {lek} bez zmian. {lek} jest do dyspozycji, ale nie łączymy go z kawą, którą pije senior. {lek} odsuwamy, poprzednio powodował splątanie.",
    "W raporcie zapisuję, rozpoczęto {lek}, obserwacja przez siedem dni. {lek} zlecony doraźnie, personel poinstruowany. {lek} czasowo wstrzymany.",
    "Proszę utrzymać {lek} codziennie o tej samej porze, najlepiej po obiedzie. {lek} zostaje jako lek na ból w gorszych dniach. {lek} na razie odkładamy, wrócimy do tematu po badaniach.",
    "Zmniejszamy ilość {lek} o jeden stopień, bo senior jest zbyt senny rano. {lek} można zastosować przed spacerem. {lek} pozostaje wstrzymany.",
    "Startujemy z {lek} na noc, proszę obserwować czy senior nie wstaje do toalety i się nie chwieje. {lek} wyłącznie doraźnie. {lek} na później, decyzja po wynikach.",
    "Mieszkaniec ma problemy ze snem, {lek} podajemy po kolacji. {lek} w razie wybudzenia w nocy. {lek} niezalecany, ryzyko upadku z łóżka.",
    "Po infekcji wracamy do {lek} w pełnej ilości, rozkruszone do jedzenia. {lek} proszę ograniczyć. {lek} wycofany z powodu zawrotów.",
    "Dziś modyfikujemy kartę, {lek} rano. {lek} wyłącznie w razie ataku paniki. {lek} pomijamy, senior był po nim agresywny.",
    "Przed zajęciami terapeutycznymi trzymamy ilość {lek} stałą. {lek} można podać przed grupą. {lek} pozostaje zawieszony.",
    "Jeśli senior ma sucho w ustach po {lek}, proszę częściej podawać picie. {lek} może to pogłębiać. {lek} pozostaje poza planem.",
    "Senior chce wychodzić sam, {lek} najlepiej podawać po powrocie ze spaceru. {lek} nie przed wyjściem. {lek} czasowo wycofany.",
    "W weekend proszę nie zmieniać pory {lek}. Apteczka oddziałowa ma zapas {lek}. {lek} w rezerwie.",
    "Dieta lekkostrawna przy {lek} jest wskazana. {lek} łączyć z wodą. {lek} wycofany.",
    "Cel to wyciszenie seniora, dlatego {lek} stale. {lek} tylko przy krzykach. {lek} odłożony.",
    "Jeżeli po {lek} senior trzyma się za brzuch, proszę podawać lek w trakcie jedzenia. {lek} może być pominięty, jeśli senior śpi. {lek} wstrzymany.",
    "Zwiększamy {lek} co trzy dni, proszę notować aktywność. {lek} rzadko. {lek} pomijamy.",
    "Po szczepieniu można kontynuować {lek}. {lek} dopuszczalny w razie bólu ręki. {lek} odsunięty.",
    "Proszę powiesić kartkę przy łóżku, {lek} o godzinie dwudziestej pierwszej. {lek} w rezerwie.",
    "Gorączka nie zmienia zlecenia na {lek}. {lek} jednorazowo przy temperaturze. {lek} pomijamy.",
    "Senior przestał zgłaszać zawroty, kontynuujemy {lek}. {lek} interwencyjnie. {lek} zostaje wycofany.",
    "Jeśli senior jest zdenerwowany wizytą, nie podawać {lek} na zapas, tylko {lek} jak trzeba. {lek} kontynuujemy. {lek} na razie nie.",
    "Po wizycie rehabilitanta nie zmieniać {lek}. {lek} przy bólu mięśni. {lek} nadal wstrzymany.",
    "Seniorzy lubią stały rytm, {lek} zawsze po obiedzie. {lek} nie łączyć z kawą zbożową. {lek} nie wraca.",
    "Jeśli są skurcze nóg, proszę zgłosić, {lek} może to nasilać. {lek} ograniczyć. {lek} odłożony.",
    "Na przepustkę do domu dać {lek} w kopercie. {lek} użyć w razie potrzeby. {lek} nie planowany.",
    "Podsumowując, {lek} stale, {lek} doraźnie, {lek} wstrzymany. Proszę o dokładne raportowanie zmian zachowania.",
    "Zlecono {lek} przed ciszą nocną, rozkruszyć do musu. {lek} dostępny w apteczce przy nagłym pobudzeniu, limit jedna tabletka na zmianę. Nie powielać po uspokojeniu.",
    "Włączyć {lek} do karty, podać po toalecie po wstaniu. {lek} interwencyjnie przy oporze, maksymalnie raz na dobę. {lek} wykreślony do wizyty geriatry, obserwować senność.",
    "Utrzymać {lek} przez dwa tygodnie dla stałego rytmu. Przy płaczliwości doraźnie {lek}, odnotować w raporcie. {lek} wycofany przez kłopoty z połykaniem, przy wysypce po {lek} wezwać lekarza.",
    "Wdrożyć {lek} od dzisiaj, start od połówki, ocena za tydzień. {lek} tylko w skrajnym niepokoju, zakaz łączenia z alkoholem. {lek} pomijamy do czasu wyników krwi. Przy zawrotach po {lek} asekurować chód.",
    "Kontynuacja {lek} według karty jako bazy uspokajającej. {lek} awaryjnie, wpisać godzinę, pilnować limitu. {lek} zawieszony, możliwy powrót po ustabilizowaniu ciśnienia. Przy wymiotach po {lek} podać na pełny żołądek.",
    "Zmiana, po wstaniu {lek} w małej ilości, na koniec dnia obserwacja, {lek} tylko przy agresji. {lek} w depozycie z powodu kołatań serca. Ból brzucha po {lek} zgłosić koordynującej, zrezygnować z podania.",
    "Wznowienie {lek} w leczeniu przewlekłym, pilnować regularności. {lek} przed rehabilitacją przy bólu. {lek} w rezerwie, powrót tylko przy braku reakcji na {lek}. Zakaz własnych ziół rodziny.",
    "Stały czas, {lek} po kolacji, {lek} w dyżurce na nagłe wypadki. {lek} zakazany, powodował upadki. Prowadzić kartę obserwacji po {lek} celem korekty.",
    "Zwiększenie {lek} o jednostkę za tydzień przy dobrej tolerancji. {lek} podać przed godziną osiemnastą, chronić rytm snu. {lek} wstrzymany do konsultacji. Przy duszności po {lek} wezwać pogotowie.",
    "Zostajemy przy {lek}, widać poprawę. {lek} przed wizytą rodziny na emocje. {lek} wycofany, powodował omdlenia. Pamiętać o pojeniu przy {lek}.",
    "Start {lek} jutro przy śniadaniu, ilość startowa przez trzy dni. {lek} tylko przy silnym bólu kręgosłupa, co osiem godzin. {lek} do decyzji po wynikach moczu. Przy {lek} wzmóc nadzór z powodu senności.",
    "Podać {lek} w małej ilości na początku zmiany dziennej z jogurtem. {lek} ratunkowo przy pobudzeniu. {lek} wstrzymany do kontroli, powrót jeśli {lek} za słaby.",
    "Przed szpitalem nie podawać {lek} przez czterdzieści osiem godzin. {lek} bez zmian, {lek} doraźnie przed snem. Po powrocie wznowić {lek}, {lek} zależnie od lekarza.",
    "Z uwagi na osłabienie {lek} minimalnie, monitorować. {lek} doraźnie przy sygnalizacji bólu. {lek} odłożony do depozytu.",
    "Sonda, {lek} rozkruszyć, podać z wodą. {lek} wyjątkowo przy temperaturze. {lek} wstrzymany z powodu zalegania treści.",
    "Wątroba, {lek} od połówki, obserwować skórę. {lek} doraźnie, maksymalnie dwa razy na dobę. {lek} wycofany, złe wyniki.",
    "Nerki, {lek} co drugi dzień przez tydzień. {lek} w razie potrzeby, wpisać w raport. {lek} niezalecany, czekać na nefrologa.",
    "Wyjazd, spakować {lek} w kasetkę. {lek} przed podróżą na nerwy. {lek} wycofany, rodzina wie.",
    "Kontrola za tydzień, codziennie {lek} o godzinie dwudziestej. {lek} przy wzroście agresji. {lek} rezerwa, powrót przy braku efektów {lek}.",
    "Modyfikacja, {lek} więcej po trzech dniach przy dobrej tolerancji. {lek} tylko wyjątkowo. {lek} na razie nie, ocena za miesiąc.",
    "Jeśli śpi po {lek} w dzień, przenieść na noc. {lek} wzmaga pragnienie, poić. {lek} wykreślony z powodu złej tolerancji.",
    "Ostrożnie, {lek} startowo przez tydzień, mierzyć ciśnienie. {lek} na pogorszenie, maksymalnie dwa razy. {lek} pomijamy z powodu upadków.",
    "Rehabilitacja, {lek} bez zmian. {lek} godzinę przed ćwiczeniami. {lek} wstrzymany, ocena za miesiąc.",
    "Bezpieczeństwo, nie łączyć {lek} z witaminami. {lek} wywołuje wymioty, stop. Przy plamach na skórze po {lek} przerwać.",
    "Zmiana pory, {lek} po śniadaniu, aktywizacja. {lek} awaryjnie, nie po kolacji. {lek} rezerwa, powrót przy bólach.",
    "Szczepienie, {lek} podajemy normalnie. {lek} dzień po przy gorączce. {lek} wstrzymany z powodu biegunki.",
    "Wybudzenia, {lek} godzinę później. {lek} raz przy nocnych krzykach. {lek} nie wraca, skupienie na {lek}.",
    "Po antybiotyku, {lek} stale, dużo pić. {lek} interwencyjnie. {lek} wstrzymany na tydzień z powodu żołądka.",
    "Serce, puls przed {lek}. {lek} podnosi ciśnienie, nie na noc. {lek} wycofany, czekać na e ka gie.",
    "Do wyników {lek} bez zmian. {lek} dostępny, nie łączyć z kawą. {lek} odsunięty z powodu splątania.",
    "Raport, start {lek}, obserwacja siedem dni. {lek} doraźnie, personel poinstruowany. {lek} wstrzymany.",
    "Utrzymać {lek} stale po obiedzie. {lek} na ból w gorsze dni. {lek} odłożony do badań.",
    "Mniej {lek}, senność po wstaniu. {lek} przed spacerem. {lek} wstrzymany.",
    "Start {lek} na noc, obserwować chód i toaletę. {lek} doraźnie. {lek} decyzja po wynikach.",
    "Problemy ze snem, {lek} po kolacji. {lek} przy wybudzeniu. {lek} ryzyko upadku, niezalecany.",
    "Po infekcji, {lek} pełna ilość, w jedzeniu. {lek} ograniczyć. {lek} wycofany z powodu zawrotów.",
    "Karta, {lek} po wstaniu. {lek} przy ataku paniki. {lek} pomijamy z powodu agresji.",
    "Terapia, {lek} stała ilość. {lek} przed grupą. {lek} zawieszony.",
    "Suchość w ustach po {lek}, częściej poić. {lek} pogłębia problem. {lek} poza planem.",
    "Spacer, {lek} po powrocie. {lek} nie przed wyjściem. {lek} wycofany.",
    "Weekend, {lek} bez zmian pory. Zapas {lek} w apteczce. {lek} rezerwa.",
    "Lekkostrawne przy {lek}. Nie łączyć {lek} z sokiem grejpfrutowym. {lek} wycofany.",
    "Wyciszenie, {lek} stale. {lek} przy krzykach. {lek} odłożony.",
    "Ból brzucha po {lek}, podać przy jedzeniu. Pominąć {lek} jak śpi. {lek} wstrzymany.",
    "Skurcze nóg, zgłosić, {lek} może nasilać. {lek} ograniczyć. {lek} odłożony.",
    "Przepustka, {lek} w kopercie. {lek} w razie potrzeby. {lek} nie planowany.",
    "Raport, {lek} stale, {lek} doraźnie, {lek} wstrzymany. Raportować zmiany.",
    "Odnotowano decyzję o powrocie do {lek} aplikowanego przed spoczynkiem nocnym, popijać wodą po konsumpcji. {lek} pełni funkcję wsparcia doraźnego, limit trzech aplikacji na dobę, z zastrzeżeniem niepowielania po ustąpieniu dyskomfortu.",
    "Plan zakłada włączenie {lek} w godzinach nocnych po wypełnieniu żołądka lekkim daniem. {lek} stanowi element interwencyjny w dniach zaostrzenia, jedna aplikacja dobowo. Decyzja o {lek} zawieszona do stabilizacji wyników, z koniecznością monitorowania senności.",
    "Utrzymanie terapii {lek} przez dwa tygodnie ma na celu adaptację. Dopuszcza się interwencyjne zastosowanie {lek} w kryzysie, ustąpienie bólu znosi konieczność aplikacji. Historia problemów gastrycznych wyklucza czasowo {lek}, a reakcje skórne obligują do zaprzestania leczenia.",
    "Inicjacja {lek} w dniu bieżącym od minimalnej ilości, z ewaluacją po siedmiu dniach. {lek} przewidziano do zastosowań ratunkowych przy ostrym bólu, zakaz łączenia z etanolem. Powrót do {lek} zależy od wyników, a zawroty głowy po {lek} są przeciwwskazaniem do kierowania.",
    "Terapia bazowa opiera się na kontynuacji {lek} bez modyfikacji. {lek} jako zabezpieczenie ratunkowe wymaga ewidencji godzin i przestrzegania limitów. Czasowa rezygnacja z {lek} zależna od normalizacji morfologii, nudności po {lek} wymagają nawadniania i przyjmowania na pełny żołądek.",
    "Strategia obejmuje poranną podaż {lek} w niskim wariancie oraz wypoczynek na koniec dnia, {lek} zarezerwowano dla silniejszego dyskomfortu. Z uwagi na kołatania serca {lek} wyłączony do przyszłego tygodnia. Ból brzucha po {lek} sygnałem do przerwania kuracji.",
    "Przywrócenie {lek} w trybie przewlekłym bez nagłego przerywania. Dopuszcza się użycie {lek} przed wysiłkiem przy narastaniu problemów. {lek} stanowi opcję rezerwową przy braku odpowiedzi na {lek}, zakaz suplementacji ziołowej.",
    "Harmonogram zakłada aplikację {lek} po ostatnim daniu, przy posiadaniu {lek} jako zabezpieczenia. Epizody senności wykluczają obecnie {lek}. Prowadzenie dziennika efektów po {lek} umożliwi korektę ilości substancji.",
    "Eskalacja ilości {lek} po tygodniu przy braku negatywnych reakcji. Zastosowanie {lek} wymaga ostrożności i przestrzegania godziny osiemnastej celem ochrony rytmu. Wznowienie {lek} po konsultacji, duszność lub obrzęki po {lek} wymagają pilnej interwencji.",
    "Poprawa funkcjonalna uzasadnia utrzymanie {lek}. {lek} może zostać wykorzystany profilaktycznie przed podróżą. Spadki ciśnienia powodem wycofania {lek}. Wskazana hydratacja oraz uwaga na obniżenie tolerancji wysiłkowej przy {lek}.",
    "Zalecenie, rozpoczęcie {lek} po przebudzeniu, utrzymanie ilości startowej przez trzy doby. Częstość stosowania {lek} do interwału ośmiogodzinnego, zasadna przy nasilonym bólu. Decyzja o {lek} po analizie laboratoryjnej, senność po {lek} wyklucza obsługę maszyn.",
    "Rozpisano {lek} w małej ilości na początek dnia, po śniadaniu, z wodą. {lek} w trybie na żądanie przy nasileniu dolegliwości, odstęp ośmiogodzinny. Wstrzymanie {lek} do kontroli, przywrócenie zależy od odpowiedzi na {lek}.",
    "Planowy zabieg wymaga zawieszenia {lek} na czterdzieści osiem godzin przed hospitalizacją. Schemat z {lek} bez zmian, {lek} dopuszczalny jednorazowo w przeddzień. Wznowienie {lek} w pierwszej dobie pooperacyjnej, losy {lek} zależą od opinii anestezjologicznej.",
    "Ciąża wymusza utrzymanie {lek} na minimalnym pułapie przy monitoringu. Zastosowanie {lek} ograniczone do wyższej konieczności. {lek} odłożono, powrót wymaga konsylium perinatologicznego.",
    "Choroba wątroby implikuje redukcję ilości {lek} o pięćdziesiąt procent i obserwację nietolerancji. Limit dobowy dla {lek}, dwie aplikacje. Podwyższone parametry wątrobowe przyczyną wycofania {lek}.",
    "Niewydolność nerek wymusza schemat alternatywny {lek} co drugi dzień. {lek} stosowany w razie potrzeby, ewidencja godzinowa. Brak wskaźnika e gie ef er uniemożliwia rekomendację {lek}.",
    "Podróż wymaga zapasu {lek} i aplikacji według czasu domowego. Zastosowanie {lek} przed wylotem uzasadnione nasileniem dolegliwości. Zgłaszane zawroty głowy skutkują utrzymaniem wycofania {lek}.",
    "Teleporada za siedem dni poprzedzona codzienną aplikacją {lek} o godzinie dwudziestej. Użycie {lek} uzasadnia ból przekraczający próg. Minimalna ilość {lek} rozważana przy braku reakcji na {lek}.",
    "Miareczkowanie {lek}, zwiększenie ilości po trzech dobach przy tolerancji. Użycie {lek} ograniczone do stanów wyjątkowych. Bilans korzyści {lek} do oceny po miesiącu, lek poza schematem.",
    "Senność po {lek} wskazaniem do przesunięcia aplikacji na noc i rezygnacji z prowadzenia. Suchość w ustach przy {lek} wymaga płynów. Zła tolerancja powodem trwałego wycofania {lek}.",
    "Wiek geriatryczny uzasadnia start {lek} od ilości początkowej i pomiary ciśnienia. {lek} stanowi zabezpieczenie, limit dwóch aplikacji. Chwiejność postawy skutkuje pominięciem {lek}.",
    "Dni obciążenia fizycznego nie uzasadniają modyfikacji ilości {lek}. Dopuszcza się podanie {lek} sześćdziesiąt minut przed treningiem przy predykcji bólu. Ocena powrotu do {lek} po miesiącu karencji.",
    "Ryzyko interakcji wyklucza łączenie {lek} z alkoholem i nowymi suplementami. Zgaga i nudności skutkują rezygnacją z {lek} do poprawy. Wysypka po {lek} wskazaniem do przerwania podaży.",
    "Przesunięcie aplikacji {lek} na godziny poranne, po śniadaniu, celem redukcji senności. {lek} pełni funkcję awaryjną, nie stosować po godzinie osiemnastej. Brak redukcji bólu przesłanką powrotu do {lek}.",
    "Szczepienie nie stanowi przeciwwskazania do {lek}. Pogorszenie samopoczucia po iniekcji umożliwia jednorazowe zastosowanie {lek}. Dolegliwości żołądkowe przyczyną wstrzymania {lek}.",
    "Nocne wybudzenia sugerują przesunięcie aplikacji {lek} o godzinę. Objawy nocne dopuszczają jednorazowe włączenie {lek}. Optymalizacja schematu opartego na {lek} priorytetem, stąd brak powrotu do {lek}.",
    "Zakończenie antybiotykoterapii umożliwia powrót do {lek} przy nawodnieniu. {lek} ma charakter interwencyjny. Kumulacja działań niepożądanych wymusza wstrzymanie {lek}.",
    "Wymagane monitorowanie tętna po włączeniu {lek}, wywiad kardiologiczny. Wzrost ciśnienia po {lek} sugeruje unikanie aplikacji przed snem. Brak e ka gie odsuwa decyzję o włączeniu {lek}.",
    "Stabilizacja parametrów krwi warunkiem utrzymania {lek}. Aktywność zawodowa dopuszcza użycie {lek}, bez kofeiny. Wpływ na koncentrację skutkuje odsunięciem {lek}.",
    "Włączono {lek} w ilości startowej, ocena tolerancji po siedmiu dniach. Przeprowadzono edukację o interakcjach i limitach dla {lek}. Warunkowy plan powrotu do {lek} zależy od odpowiedzi klinicznej.",
    "Rekomendowana stała pora przyjmowania {lek} po głównym daniu. {lek} do przełamywania bólu w gorszych dniach z ewidencją. Kontrola enzymów niezbędna do ponownego rozważenia {lek}.",
    "Redukcja ilości {lek} o poziom wynika z porannej senności. Przed aktywnością dopuszcza się {lek}, limit trzech aplikacji. Epizody kołatania serca uzasadniają wstrzymanie {lek}.",
    "Schemat zakłada aplikację {lek} przed snem, zakaz prowadzenia w pierwszej dobie. Brak poprawy godzinę po {lek} nie upoważnia do powtórzenia. Wyniki elektrolitów i e ka gie zadecydują o {lek}.",
    "Praca zmianowa wymaga przyjmowania {lek} po aktywności zawodowej. Zmiana nocna dopuszcza posiadanie porcji {lek}. Ryzyko sedacji wyklucza stosowanie {lek}.",
    "Przebyta infekcja pozwala na powrót do pełnej ilości {lek}, osłona posiłkowa. Sytuacje wyjątkowe dopuszczają użycie {lek}. Stabilizacja pozwoli rozważyć niższy wariant {lek}, wycofany przez zawroty.",
    "Modyfikacja uwzględnia {lek} po przebudzeniu i kontrolę za tydzień. Nagłe pogorszenie uzasadnia użycie {lek} przed godziną dziewiętnastą. Zaburzenia koncentracji powodem wyeliminowania {lek}.",
    "Aktywność sportowa wymaga utrzymania stałej ilości {lek}. Podaż {lek} czterdzieści pięć minut przed wysiłkiem dopuszczalna, dbać o nawodnienie. Istotne efekty uboczne przyczyną zawieszenia {lek}.",
    "Zwiększenie podaży płynów zalecane przy suchości po {lek}. Oszczędne stosowanie {lek} wynika z ryzyka nasilenia objawu. Do wizyty {lek} wyłączony z farmakoterapii.",
    "Prowadzenie pojazdów przesuwa stosowanie {lek} na czas po pracy, ocena tolerancji. Zachowanie sześciogodzinnego odstępu od prowadzenia konieczne przy {lek}. Testy uwagi zadecydują o powrocie do {lek}.",
    "Weekend nie zwalnia z zachowania stałej pory dla {lek}. Apteczka podręczna powinna zawierać {lek}. Nieskuteczność {lek} przesłanką do sięgnięcia po rezerwowy {lek}.",
    "Dieta wysokobiałkowa przed {lek} poprawia tolerancję. Ryzyko kołatań wyklucza łączenie {lek} z energetykami. Normalizacja magnezu i potasu warunkiem powrotu do {lek}.",
    "Cel terapeutyczny, redukcja problemów o pięćdziesiąt procent, wymaga stabilnej podaży {lek}. Profilaktyka {lek} jest błędem, lek służy interwencji. Niekorzystny bilans podtrzymuje decyzję o odłożeniu {lek}.",
    "Ból brzucha po {lek} wymaga przyjęcia na treści pokarmowej, popić wodą. Ustąpienie problemów pozwala pominąć {lek}. Ocena gastrologiczna niezbędna do wznowienia {lek}.",
    "Stopniowa eskalacja {lek} co trzy dni wymaga monitorowania. Ochrona rytmu dobowego sugeruje unikanie {lek} przed snem. Nieakceptowalne reakcje wykluczają {lek}.",
    "Lot wymaga zapasu {lek} i stosowania według czasu zamieszkania. Nasilenie problemów po lądowaniu dopuszcza użycie {lek}. Podróż nie przewiduje wznowienia {lek}.",
    "Szczepienie nie koliduje z terapią {lek}. Pogorszenie stanu dzień po szczepieniu umożliwia aplikację {lek}. Męczliwość powodem odsunięcia {lek}.",
    "Zalecana rozpiska z godziną dwudziestą pierwszą dla {lek}. Dziennik powinien zawierać godzinę i efekt po {lek}. Wyniki laboratoryjne zadecydują o użyciu {lek} z rezerwy.",
    "Gorączka nie uzasadnia zmiany ilości {lek}. Jednorazowe użycie {lek} w dobie dopuszczalne przy nasileniu bólu. Wyzdrowienie warunkiem powrotu do {lek}.",
    "Ustąpienie zawrotów po zmianie pory pozwala na kontynuację {lek} na noc. Dublowanie porcji {lek} zabronione. Brak wskazań podtrzymuje wycofanie {lek}.",
    "Dopisano adnotację o kontynuacji {lek} w stałej ilości. Edukacja o limitach {lek} zrealizowana. Plan kontroli za dziesięć dni dotyczy wstrzymanego {lek}.",
    "Stres w pracy nie jest wskazaniem do profilaktyki {lek}, stosować przy realnym wzroście problemów. Kluczem do sukcesu {lek} jest regularność. Pogorszenie tolerancji wyklucza obecnie {lek}.",
    "Kinezyterapia podlega ocenie niezależnej od {lek}, brak modyfikacji. Zaostrzenie po ćwiczeniach dopuszcza {lek}. Brak wskazań do terapii łączonej utrzymuje wstrzymanie {lek}.",
    "Nocna toaleta i woda przy łóżku ułatwiają tolerancję {lek}. Zakaz przyjmowania {lek} między północą a godziną szóstą rano. Święta terminem rewizji decyzji o {lek} rezerwowym.",
    "Zmiana producenta {lek} wymaga trzydniowej obserwacji. Interwencyjny charakter {lek} bez zmian. Brak historycznej poprawy uzasadnia wycofanie {lek}.",
    "Praca zmianowa wymaga spożycia {lek} po głównym daniu, niezależnie od zegara. Łączenie {lek} z kofeiną przeciwwskazane. Ryzyko senności wyklucza powrót {lek}.",
    "Zwiększenie podaży elektrolitów zalecane przy skurczach po {lek}. Ograniczenie {lek} wynika z ryzyka nasilenia zjawiska. Monitoring trwa przy odłożonym {lek}.",
    "Wyjazd służbowy nie zwalnia ze stosowania {lek} według schematu. Przekroczenie typowej intensywności problemów pozwala na użycie {lek}. Wyjazd nie uwzględnia planu dla {lek}.",
    "Podsumowanie, {lek} stanowi fundament terapii, {lek} pełni funkcję interwencyjną z limitem, a {lek} pozostaje zawieszony do oceny klinicznej. Dziennik samopoczucia zwiększy precyzję decyzji."
]

sd.default.latency = "low"


# =========================
# LOGIKA GUI I NAGRYWANIA
# =========================
class AudioRecorderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Rejestrator Zdań Medycznych")
        self.root.geometry("850x400")  # Lekko poszerzyłem okno, aby zmieścił się nowy przycisk
        self.root.configure(padx=20, pady=20)

        # Inicjalizacja katalogów i liczników
        self.lek_dirs = {}
        self.idx_by_med = {}
        for lek in MEDICINES:
            d = OUT_ROOT / lek.replace(" ", "_")
            d.mkdir(parents=True, exist_ok=True)
            self.lek_dirs[lek] = d
            self.idx_by_med[lek] = self.get_next_index(d)

        self.med_cycle = itertools.cycle(MEDICINES)

        # Stan wewnętrzny
        self.stream = None
        self.bytebuf = bytearray()
        self.is_recording = False
        self.is_finishing = False
        self.current_lek = ""
        self.current_text = ""
        self.current_idx = 0

        self.setup_ui()
        self.load_next_sentence()

        # Zdarzenia klawiatury
        self.root.bind('<space>', self.toggle_pause)
        self.root.bind('<Return>', self.finish_recording)
        self.root.bind('<r>', self.reset_recording)
        self.root.bind('<R>', self.reset_recording)

        # Zabezpieczenie przed zamknięciem okna podczas nagrywania
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def setup_ui(self):
        self.info_label = tk.Label(self.root, text="Przygotowywanie...", font=("Arial", 14), fg="gray")
        self.info_label.pack(pady=(0, 10))

        self.text_label = tk.Label(self.root, text="", font=("Arial", 16, "bold"), wraplength=750, justify="center")
        self.text_label.pack(expand=True, fill="both", pady=20)

        self.status_label = tk.Label(self.root, text="Oczekiwanie...", font=("Arial", 14, "italic"))
        self.status_label.pack(pady=10)

        # Ramka na przyciski
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)

        self.btn_toggle = tk.Button(btn_frame, text="Pauza / Wznów (Spacja)", width=22, font=("Arial", 12),
                                    command=self.toggle_pause)
        self.btn_toggle.pack(side="left", padx=5)

        self.btn_reset = tk.Button(btn_frame, text="Resetuj zdanie (R)", width=22, font=("Arial", 12),
                                   command=self.reset_recording)
        self.btn_reset.pack(side="left", padx=5)

        self.btn_finish = tk.Button(btn_frame, text="Zapisz nagrane zdanie (Enter)", width=26, font=("Arial", 12),
                                    command=self.finish_recording)
        self.btn_finish.pack(side="left", padx=5)

    def get_next_index(self, folder: Path) -> int:
        idx = 1
        pattern = str(folder / "sentence_*.wav")
        existing = [Path(p).name for p in glob.glob(pattern)]
        if existing:
            nums = []
            for name in existing:
                m = re.match(r"sentence_(\d{4})__.*\.wav$", name)
                if m:
                    nums.append(int(m.group(1)))
            if nums:
                idx = max(nums) + 1
        return idx

    def load_next_sentence(self):
        self.bytebuf = bytearray()
        self.is_finishing = False
        self.current_lek = next(self.med_cycle)

        template = random.choice(LongSentenceValidation_v2)
        self.current_text = template.format(lek=self.current_lek)
        self.current_idx = self.idx_by_med[self.current_lek]

        self.info_label.config(
            text=f"Lek: {self.current_lek} (Ilość nagrań tej nazwy: {self.current_idx}) | Seed: {SEED}")
        self.text_label.config(text=self.current_text)

        # Automatyczne rozpoczęcie nagrywania nowego zdania
        self.start_stream()
        self.is_recording = True
        self.update_status_ui()

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        self.bytebuf.extend(indata.tobytes())

    def start_stream(self):
        if self.stream is None:
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                blocksize=BLOCKSIZE,
                callback=self.audio_callback,
            )
            self.stream.start()

    def stop_stream(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def update_status_ui(self):
        if self.is_recording:
            self.status_label.config(text="NAGRYWAM! (Enter = Zapisz | R = Resetuj | Spacja = Pauza)", fg="red")
        else:
            self.status_label.config(text="PAUZA. (Spacja = wznowienie | Enter = Zapisz | R = Resetuj)", fg="orange")

    def toggle_pause(self, event=None):
        if self.is_finishing: return

        if self.is_recording:
            self.stop_stream()
            self.is_recording = False
        else:
            self.start_stream()
            self.is_recording = True

        self.update_status_ui()

    def reset_recording(self, event=None):
        """Dodana funkcja resetująca aktualne zdanie od nowa"""
        if self.is_finishing: return

        self.stop_stream()
        self.bytebuf = bytearray()  # Wyczyszczenie dotychczasowego bufora audio
        self.start_stream()
        self.is_recording = True
        self.update_status_ui()

    def finish_recording(self, event=None):
        if self.is_finishing: return
        self.is_finishing = True
        self.status_label.config(text="Zapisywanie nagrania (post-roll)...", fg="blue")

        # Upewniamy się, że strumień jest aktywny, aby nagrać końcówkę (post-roll)
        if not self.is_recording:
            self.start_stream()

        # Obliczenie czasu oczekiwania bez blokowania głównego wątku GUI
        delay_ms = int((POSTROLL_SEC + (BLOCKSIZE / SAMPLE_RATE)) * 1000)
        self.root.after(delay_ms, self.save_and_advance)

    def save_and_advance(self):
        self.stop_stream()

        if len(self.bytebuf) == 0:
            samples = np.zeros(0, dtype=np.int16)
        else:
            samples = np.frombuffer(self.bytebuf, dtype=np.int16)

        lek_dir = self.lek_dirs[self.current_lek]
        transcript_path = lek_dir / "transcript_sentences.txt"
        safe_lek_name = self.current_lek.replace(" ", "_")

        # Upewnienie się, że nazwa pliku nie istnieje
        while True:
            fname = f"sentence_{self.current_idx:04d}____{VOICE_TAG}__{safe_lek_name}.wav"
            out_wav = lek_dir / fname
            if not out_wav.exists():
                break
            self.current_idx += 1

        # Zapis WAV
        with wave.open(str(out_wav), "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(samples.tobytes())

        # Zapis do transkryptu
        with open(transcript_path, "a", encoding="utf-8") as tf:
            tf.write(f"{fname}|{self.current_text}\n")

        # Inkrementacja indeksu dla tego leku
        self.idx_by_med[self.current_lek] = self.current_idx + 1

        # Przejście do kolejnego zdania
        self.load_next_sentence()

    def on_close(self):
        self.stop_stream()
        self.root.destroy()
        sys.exit(0)


if __name__ == "__main__":
    root = tk.Tk()
    app = AudioRecorderGUI(root)
    root.mainloop()