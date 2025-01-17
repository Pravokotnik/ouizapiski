# Umetna inteligenca, uvod v strojno učenje

#### Kaj je umetna inteligenca?
+ **Cilj:** razumeti in zgraditi inteligentne sisteme na osnovi človeškega razmišljanja, sklepanja, učenja, komuniciranja
+ vse kar dela človek ni nujno inteligentno - ali modelirati človeka ali ideal? (princip racionalnosti/optimalnosti)

#### Turingov test
Je **praktični preizkus**, ki ga je predlagal Turing, za testiranje, ali je sistem dosegel stopnjo inteligence, primerljivo s človekom. Računalnik preizkus opravi, če človeški izpraševalec po računalnikovih odgovorih ne more določiti, ali je odgovarjal človek, ali računalnik. Problem testa je v tem, da ga ni možno reproducirati ali podvreči matematični analizi.

#### Osnove umetne inteligence
+ strojno učenje
+ reševanje problemov
+ planiranje, razporejanje opravil
+ sklepanje

# Strojno učenje

#### Kaj je?
Področje umetne inteligence, ki raziskuje, kako se lahko **algoritmi samodejno izboljšujejo ob pridobivanju izkušenj**. To je potrebno, ker ni vedno možno predvideti vseh problemskih situacij ali sprememb, ali pa preprosto ni mogoče sprogramirati vsega znanja.

#### Vrste učenja
* **Nadzorovano učenje:** učni primeri podani kot vrednosti vhodov in izhodov &rarr; funkcija, ki preslika vhode v izhode (odločitveno drevo)
* **Nenadzorovano učenje:** učni primeri niso označeni - ni ciljne spremenljivke &rarr; vzorci v podatkih (gručenje)
* **Spodbujevano učenje:** inteligentni agent se uči iz zaporedja nagrad in kazni

#### Nadzorovano učenje
+ **Podana**: množica učnih primerov $(x_1,y_1), ..., (x_N,y_N)$, kjer je vsak $y_i$ vrednosti neznane funkcije $y=f(x)$
+ **Naloga**: iščemo funkcijo $h$, ki je najboljši približek $f$
+ poimenovanje: $x_i$ so atributi, $h$ je hipoteza

* imamo 2 vrsti problemov:
    + **Klasifikacijski problem:** $y_i$ diskretna spremenljivka
    + **Regresijski problem:** $y_i$ zvezna spremenljivka

>**Klasifikacija:**
>+ $y$ je diskreten (končen nabor vrednosti)
>+ $y$ se imenuje razred

>**Regresija:**
>+ $y$ je zvezen (neko število)
>+ $y$ se imenuje označba

* **Hipoteze:**
    + dobra hipoteza je dovolj **splošna** &rarr; pravilno napoveduje vrednost $y$ za nove primere
    + princip **Ockhamove britve** za izbor primerne hipoteze &rarr; najbolj preprosta
* **Prostor hipotez:**
    + binarna klasifikacija z $n$ atributi &rarr; $2^n$ različnih učnih primerov in $2^{2^n}$ hipotez
    + potrebni algoritmi za gradnjo dobrih hipotez in metode za ocenjevanje hipotez/učenja
* **Evalviranje hipotez:**
    + konsistentnost hipotez z učnimi primeri
    + splošnost (točnost za nevidene primere)
    + razumljivost

>Klasifikacijska točnost:
>$$ CA = \frac{TP + TN}{TP + TN + FP + FN} = \frac{TP + TN}{N} $$
>&nbsp;

#### Učenje odločitvenih dreves

+ **Odločitveno drevo:**
    + ponazarja relacijo med atributi (vhodne vrednosti) in ciljno spremenljivko (odločitev)
    + cilj zgraditi čim **manjše** drevo, **konsistentno** z učnimi podatki
    + **TDIDT:** hevristični požrešni algoritem, razveji in omeji &rarr; izbere najbolj pomemben atribut, rekurzivno ponavlja za poddrevesa
+ Izbor najbolj pomembnega atributa:
>    + **entropija:** $$H = -\sum_k p_k \log_2 p_k$$
>    + **informacijski prispevek:** $$IG(A) = I - I_{res}(A)$$
>$$ I_{\text{res}} = -\sum_{v_i \in A} p_{v_i} \sum_c p(c \mid v_i) \log_2 p(c \mid v_i)$$
>    + **information gain ratio:** $$GR(A) = \frac{IG(A)}{I(A)}$$
>    + **gini index:** $$\text{Gini}(A) = \sum_v p(v) \sum_{c_1 \neq c_2} p(c_1 \mid v) p(c_2 \mid v)$$
+ Težava z večvrednostmi atributi &rarr; relativni informacijski prispevek ($GR$), alternativne mere ($Gini$), binarizacija atributov (višja $CA$)
+ Kratkovidnost TDIDT: najboljši atribut izbira lokalno
+ Privzeta točnost: uporabljamo verjetnost večinskega razreda &rarr; drevo je uporabno, če je njegova točnost višja od privzete
+ Pristranost na učni množici: lahko pride do pretiranega prilagajanja na učni množici &rarr; podatke zato delimo na učno in testno množico ($70-30$)

#### Učenje dreves iz šumnih podatkov

Možne težave zaradi nepopolnih podatkov:
+ učenje šuma in ne dejanske funkcije, ki generira podatke
+ pretirano prilagajanje vodi v prevelika drevesa
+ slaba razumljivost dreves
+ nižja $CA$ na novih podatkih

Rešitev za nastale težave je **rezanje dreves**. Ideja za rešitev je, da nižji deli drevesa predstavljajo večje lokalno prilagajanje učnim podatkom. Zato drevo režemo, da dosežemo večjo posplošitev.

Rezanje dreves:
+ **Rezanje vnaprej**:
    + dodatni kriterij za zaustavitev gradnje drevesa, hitro, kratkovidno
+ **Rezanje nazaj**:
    + po gradnji celotnega drevesa odstranimo manj zanseljive dele, počasno, upošteva informacijo celega drevesa
>    + **REP - rezanje z zmanjševanjem napake:**
>        + uporaba rezalne možice
>        + delitev podatkov: učna množica (70%, od tega 70% za gradnjo, 30% rezalna), testna množica (30%)
>    + **MEP - rezanje z minimizacijo napake:**
>        + za vsako vozlišče izračunamo statično napako ($e$), in vzvratno napako ($E$)
>        + režemo, če je statična napaka manjša od vzvratne ($E \geq e$)

Ocenjevanje verjetnosti (ocena napake v vozlišču):
>* **Laplaceova ocena:**
>    + ne upošteva apriorne verjetnosti
>    $$p = \frac{n + 1}{N + k}$$
>* **m-ocena:** 
>    + manj kot je šuma, manjši je $m$
>    + posplošitev Laplaceove za $m=k$, $p_a = \frac{1}{k}$
>    $$p = \frac{n + p_am}{N + m}$$

#### Ocenjevanje učenja

Hipoteze ocenjujemo glede na njihovo točnost, razumljivost in/ali kompleksnost. Točnost lahko ocenjujemo na učnih, testnih (intervali zaupanja) ali novih podatkih. Za uspešno učenje in za zanesljivo ocenjevanje rabimo čim več podatkov, kar si nasprotuje. Možni rešitvi sta izločevanje testne množice kadar je učnih podatkov dovolj, ali večkratne delitve na učno in testno množico.

**k-kratno prečno preverjanje:**
+ celo učno množico razdelimo na $k$ disjunktnih množic
+ za vsako disjunktno množico, to množico uporabimo kot **testno**, preostalih $k-1$ množic pa kot **učno**
+ vplive izbranega razbitja na podmnožice zmanjšamo z večkratnim ponavljanjem preverjanja
+ metoda **izloći enega:** $k$ je enak številu primerov, najbolj stabilno, časovno zelo zamudno

#### Vrste atributov, diskretizacija, obravnava manjkajočih vrednosti

Zvezne ali numerične atribute načeloma diskretiziramo z intervali, ki **maksimizirajo informacijski prispevek**.

Obravnava **manjkajočih vrednosti atributov** vključuje različne pristope, kot so učenje z manjkajočimi vrednostmi, ignoriranje primerov z neznanimi vrednostmi, uporaba posebne oznake (NA/UNKNOWN), nadomeščanje manjkajočih vrednosti (npr. z povprečno, najbolj pogosto ali napovedano vrednostjo)...

#### Naivni Bayesov klasifikator

>**Bayesovo pravilo:**
>$$P(B|A) = \frac{P(A|B) \cdot P(B)}{P(A)}$$
>
>**Naivni Bayes:**
>$$P(C \mid X_1, X_2, \dots, X_n) = \frac{P(C) \cdot P(X_1, X_2, \dots, X_n \mid C)}{P(X_1, X_2, \dots, X_n)} $$
>$$P(C \mid X_1, X_2, \ldots, X_n) \approx \frac{P(C) \cdot \prod_{i} P(X_i \mid C)}{\prod_{i} P(X_i)}$$
>(približki veljajo le, le so atributi med sabo dovolj neodvisni)
>&nbsp;
>**Bayesov klasifikator:** primer klasificiramo v razred, ki je najbolj verjeten
>$$h(C_k \mid X_1, X_2, \ldots, X_n) = \arg \max_{C_k} \left( P(C_k) \prod_{i=1}^n P(X_i \mid C_k) \right)$$
>&nbsp;

#### Nomogrami

Nomogram je pristop za vizualizacijo naivnega Bayesovega modela. Prikazuje pomembnost posameznih vrednosti vsakega atributa na ciljni razred. Vsaka vrednost atributa doprinaša določeno število točk k ciljnemu razredu. 

>**Skupno število točk:**
>$$\text{tocke}(C \mid X_1, X_2, \ldots, X_n) = \sum_i \log \frac{P(X_i \mid C)}{P(X_i \mid \overline{C})}$$
>$$\frac{P(C \mid X_i)}{P(\overline{C} \mid X_i)} = \frac{\frac{P(C \mid X_i)}{P(C)}}{\frac{P(\overline{C} \mid X_i)}{P(\overline{C})}}$$
>(izračun)

#### Metoda $k$ najbližjih sosedov

