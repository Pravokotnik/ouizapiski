# Umetina inteligenca, uvod v strojno učenje

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

**Klasifikacija:**
+ $y$ je diskreten (končen nabor vrednosti)
+ $y$ se imenuje razred

**Regresija:**
+ $y$ je zvezen (neko število)
+ $y$ se imenuje označba