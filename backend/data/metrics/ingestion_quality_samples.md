# Controle qualite de l'ingestion - echantillons reels

Genere le 2026-08-01T08:02:37+00:00 par le pipeline d'ingestion v1.1.0.

Tous les extraits ci-dessous sont copies tels quels depuis le corpus produit.

---

## 1. Pages en double colonne

Le defaut redoute est l'alternation d'une ligne de la colonne gauche et d'une
ligne de la colonne droite. La mesure objective est le **nombre de basculements
gauche/droite** dans l'ordre de lecture : en lecture correcte il vaut au plus le
nombre de bandes horizontales de la page (typiquement 1), alors que l'ordre naif
trie par ordonnee bascule a chaque paragraphe.

### Chapitre 3 - article publie - page PDF 55 (page these 40)

- Mode detecte : `double_column`
- Blocs colonne gauche / droite : 11 / 10
- Bandes horizontales : 1
- Largeur du bloc le plus large / largeur de page : **0.405**
- **Basculements avec l'ordre retenu : 1**
- Basculements avec l'ordre naif (temoin) : 11
- Colonnes non melangees : OUI

**Extrait brut (ordre de lecture reconstruit, avant nettoyage) :**

```
4
C. Cadiou and P. Yiou: Cold winters of the beginning of the 21st century in France

2
Data and methods

2.1
Data

Daily mean surface temperature (TG) data were obtained
from the ﬁfth version (ERA5) of the atmospheric reanalysis
of the European Centre for Medium-Range Weather Fore-
casts (ECMWF) (Hersbach et al., 2020). Data from 1950 to
2021 have been retrieved with a spatial resolution of 0.25°×
0.25°. Daily temperature ﬁelds have been averaged over the
smallest spatial domain including metropolitan France (42–
52° N, 5° W–9° E). ERA5 was chosen for its large time cov-
erage and its high horizontal resolution of 0.25°.

We compute running averages of temperature for four
event durations (r = 3, 10, 30 and 90 d) and determine the
minimum value for each winter (from December to Febru-
ary). This corresponds to identifying the coldest r d period
for each year, or TGrd. For an even value
```

**Extrait nettoye :**

```
2 Data and methods

2.1 Data

Daily mean surface temperature (TG) data were obtained from the fifth version (ERA5) of the atmospheric reanalysis of the European Centre for Medium-Range Weather Forecasts (ECMWF) (Hersbach et al., 2020). Data from 1950 to 2021 have been retrieved with a spatial resolution of 0.25°× 0.25°. Daily temperature fields have been averaged over the smallest spatial domain including metropolitan France (42- 52° N, 5° W-9° E). ERA5 was chosen for its large time coverage and its high horizontal resolution of 0.25°.

We compute running averages of temperature for four event durations (r = 3, 10, 30 and 90 d) and determine the minimum value for each winter (from December to February). This corresponds to identifying the coldest r d period for each year, or TGrd. For an even value of r, we compute the minimum over all time steps t:

(r/2)-1 X

TGrdt = 1

TGt+i, (1)

r
```

**Justification :** l'ordre retenu emet l'integralite de la colonne gauche avant la colonne droite dans chaque bande, d'ou un nombre de basculements egal a 1 pour 1 bande(s). L'ordre naif en produirait 11, soit autant de phrases coupees par un fragment de l'autre colonne.

### Chapitre 4 - article en revision - page PDF 89 (page these 74)

- Mode detecte : `single_column`
- Blocs colonne gauche / droite : 3 / 0
- Bandes horizontales : 1
- Largeur du bloc le plus large / largeur de page : **0.876**
- **Basculements avec l'ordre retenu : 0**
- Basculements avec l'ordre naif (temoin) : 0
- Colonnes non melangees : OUI

> **Constat mesure : cette plage ne contient aucune page en double colonne.** Le bloc le plus large occupe 0.876 de la largeur de page, ce qui correspond a une composition pleine largeur. Le risque de melange de colonnes ne s'applique donc pas ici. Aucun echantillon n'a ete force.

**Extrait brut (ordre de lecture reconstruit, avant nettoyage) :**

```
In France, the historical cold spell of January 1985 led to peaks in electricity demand and power outages (Caud and Vautard,
25
2018; Le Monde, 1985; RTE, 2021). More recently, the cold spell of February 2012 caused a record consumption of electricity
and put the energy network at risk (Le Monde, 2012a, b). In 2023, the electricity transmission system operator Réseau de
Transport d’Électricité (RTE) warned ahead of the winter season of several reasons for tension in the electricity supply — lack
of gas supply due to the economic and geopolitical context combined with the unavailability of parts of the nuclear power plants
for technical reasons — potentially leading to outages in the event of high demand caused by a cold spell (RTE, 2023b). The
30
resulting reduction in electricity consumption and the overall mild winter allowed avoiding power outages in France. However,
RTE stated that u
```

**Extrait nettoye :**

```
In France, the historical cold spell of January 1985 led to peaks in electricity demand and power outages (Caud and Vautard, 2018; Le Monde, 1985; RTE, 2021). More recently, the cold spell of February 2012 caused a record consumption of electricity and put the energy network at risk (Le Monde, 2012a, b). In 2023, the electricity transmission system operator Réseau de Transport d'Électricité (RTE) warned ahead of the winter season of several reasons for tension in the electricity supply - lack of gas supply due to the economic and geopolitical context combined with the unavailability of parts of the nuclear power plants for technical reasons - potentially leading to outages in the event of high demand caused by a cold spell (RTE, 2023b). The resulting reduction in electricity consumption and the overall mild winter allowed avoiding power outages in France. However, RTE stated that up to 1
```

**Justification :** page mono-colonne, lue de haut en bas. Les 0 basculements mesures confirment l'absence de structure en colonnes.

### Annexe C - supplement - page PDF 175 (page these 160)

- Mode detecte : `single_column`
- Blocs colonne gauche / droite : 6 / 0
- Bandes horizontales : 1
- Largeur du bloc le plus large / largeur de page : **0.844**
- **Basculements avec l'ordre retenu : 0**
- Basculements avec l'ordre naif (temoin) : 0
- Colonnes non melangees : OUI

> **Constat mesure : cette plage ne contient aucune page en double colonne.** Le bloc le plus large occupe 0.844 de la largeur de page, ce qui correspond a une composition pleine largeur. Le risque de melange de colonnes ne s'applique donc pas ici. Aucun echantillon n'a ete force.

**Extrait brut (ordre de lecture reconstruit, avant nettoyage) :**

```
Intensity and dynamics of extreme cold spells of the 21st century in
France from CMIP6 data: Supplementary Material.

Camille Cadiou1 and Pascal Yiou1

1Laboratoire des Sciences du Climat et de l’Environnement, UMR 8212 CEA-CNRS-UVSQ, IPSL and U Paris Saclay, 91191
Gif-sur-Yvette CEDEX, France

Correspondence: Camille Cadiou (camille.cadiou@lsce.ipsl.fr)

1
Introduction

This document contains the Z500 anomaly and standard deviation maps of the 10 CMIP6 models used with KACE-1-0-G for
SWG simulations of extreme cold spells in the article Intensity and dynamics of extreme cold spells of the 21st century in
France from CMIP6 data: BCC-CSM2-MR, CanESM5, CESM2-WACCM, CNRM-ESM2-1, EC-Earth3, FGOALS-g3, IPSL-
CM6A-LR, MPI-ESM1-2-LR, MRI-ESM2-0 and NorESM2-LM.
5

2
Z500 anomaly maps

1
```

**Extrait nettoye :**

```
Intensity and dynamics of extreme cold spells of the 21st century in France from CMIP6 data: Supplementary Material.

Camille Cadiou1 and Pascal Yiou1

1Laboratoire des Sciences du Climat et de l'Environnement, UMR 8212 CEA-CNRS-UVSQ, IPSL and U Paris Saclay, 91191 Gif-sur-Yvette CEDEX, France

Correspondence: Camille Cadiou (camille.cadiou@lsce.ipsl.fr)

1 Introduction

This document contains the Z500 anomaly and standard deviation maps of the 10 CMIP6 models used with KACE-1-0-G for SWG simulations of extreme cold spells in the article Intensity and dynamics of extreme cold spells of the 21st century in France from CMIP6 data: BCC-CSM2-MR, CanESM5, CESM2-WACCM, CNRM-ESM2-1, EC-Earth3, FGOALS-g3, IPSL-CM6A-LR, MPI-ESM1-2-LR, MRI-ESM2-0 and NorESM2-LM.

2 Z500 anomaly maps
```

**Justification :** page mono-colonne, lue de haut en bas. Les 0 basculements mesures confirment l'absence de structure en colonnes.

### Annexe B - hors sujet (temoin) - page PDF 163 (page these 148)

- Mode detecte : `double_column`
- Blocs colonne gauche / droite : 17 / 15
- Bandes horizontales : 1
- Largeur du bloc le plus large / largeur de page : **0.401**
- **Basculements avec l'ordre retenu : 1**
- Basculements avec l'ordre naif (temoin) : 24
- Colonnes non melangees : OUI

**Extrait brut (ordre de lecture reconstruit, avant nettoyage) :**

```
C. Cadiou et al.

Freitas, A.C.M., Freitas, J.M., Todd, M.: Hitting time statistics and

extreme value theory. Probab. Theory Relat. Fields 147(3-4),
675–710 (2010)
Grainger, S., Fawcett, R., Trewin, B., et al.: Estimating the uncertainty

of australian area-average temperature anomalies. Int. J. Climatol.
42(5), 2815–2834 (2022)
Hawkins, E., Sutton, R.: Time of emergence of climate signals.

Geophys. Res. Lett. 39(1) (2012)
Head, L., Adams, M., McGregor, H.V., et al.: Climate change and

australia. Wiley Interdiscip. Rev. Clim. Chang. 5(2), 175–197
(2014)
Hirabayashi, Y., Mahendran, R., Koirala, S., et al.: Global flood risk

under climate change. Nat. Clim. Chang. 3(9), 816–821 (2013)
Hobday, A.J., Pecl, G.T., Fulton, B., et al.: Climate change impacts,

vulnerabilities and adaptations. Australian marine fisheries (2018)
Holland, M.M., Smith, J.A., Everett, J.D., et al.: Latitudinal pa
```

**Extrait nettoye :**

```
Freitas, A.C.M., Freitas, J.M., Todd, M.: Hitting time statistics and

extreme value theory. Probab. Theory Relat. Fields 147(3-4), 675-710 (2010) Grainger, S., Fawcett, R., Trewin, B., et al.: Estimating the uncertainty

of australian area-average temperature anomalies. Int. J. Climatol. 42(5), 2815-2834 (2022) Hawkins, E., Sutton, R.: Time of emergence of climate signals.

Geophys. Res. Lett. 39(1) (2012) Head, L., Adams, M., McGregor, H.V., et al.: Climate change and

australia. Wiley Interdiscip. Rev. Clim. Chang. 5(2), 175-197 (2014) Hirabayashi, Y., Mahendran, R., Koirala, S., et al.: Global flood risk

under climate change. Nat. Clim. Chang. 3(9), 816-821 (2013) Hobday, A.J., Pecl, G.T., Fulton, B., et al.: Climate change impacts,

vulnerabilities and adaptations. Australian marine fisheries (2018) Holland, M.M., Smith, J.A., Everett, J.D., et al.: Latitudinal patterns in

trophic
```

**Justification :** l'ordre retenu emet l'integralite de la colonne gauche avant la colonne droite dans chaque bande, d'ou un nombre de basculements egal a 1 pour 1 bande(s). L'ordre naif en produirait 24, soit autant de phrases coupees par un fragment de l'autre colonne.

---

## 2. Reparation des cesures de fin de ligne

### Page PDF 6

- Avant : `émis-\nsions`
- Apres : `émissions`
- Contexte nettoye : `... aujourd'hui communément admis par la communauté scientifique que les émissions anthropiques de gaz à effet de serre et les changements d'us...`

### Page PDF 7

- Avant : `simu-\nlation`
- Apres : `simulation`
- Contexte nettoye : `...gé en SWG pour Stochastic Weather Generator) se comporte-t-il dans la simulation d'événements de froid extrême, et comment se compare-t-il à...`

### Page PDF 8

- Avant : `exam-\nine`
- Apres : `examine`
- Contexte nettoye : `...Le chapitre 4 examine l'avenir des vagues de froid sous différents scénarios de conc...`

---

## 3. En-tetes et pieds de page courants supprimes

Ils ont ete reperes par leur repetition mesuree a travers les pages, et non par une liste ecrite a la main.

- `#.#. extreme cold events`
- `#.#. stochastic weather generator based on analogues of circulation`
- `bibliography`
- `c. cadiou and p. yiou: cold winters of the beginning of the #st century in france`
- `c. cadiou et al.`
- `challenges in attributing the # australian rain bomb to climate change`
- `chapter #.`
- `chapter #. analogues-stochastic weather generator`
- `https://doi.org/#.#/wcd-#-#-#`
- `korean meteorological society`
- `npj climate and atmospheric science (#) #`
- `p. yiou et al.`
- `published in partnership with ceccr at king abdulaziz university`
- `s. sippel et al.: the possibility of an extremely cold central european winter`
- `weather clim. dynam., #, #-#, #`

_Total : 15 motifs repetitifs identifies._

---

## 4. Normalisation Unicode et ligatures

- Page PDF 6 : `aujourd’hui` (U+2019)
- Page PDF 7 : `l’effet` (U+2019)
- Page PDF 8 : `l’avenir` (U+2019)

---

## 5. Pages portant une numerotation etrangere

Les articles de revue inseres impriment leur PROPRE numero de page. La numerotation de la these est conservee dans `page_these` pour que les citations restent verifiables, et le numero etranger est archive dans `page_printed_detected`.

Nombre de pages concernees : **35**

| page_pdf | page_these (regle) | numero imprime | chapitre |
|---|---|---|---|
| 53 | 38 | 2 | 3 |
| 54 | 39 | 3 | 3 |
| 55 | 40 | 4 | 3 |
| 56 | 41 | 5 | 3 |
| 57 | 42 | 6 | 3 |
| 58 | 43 | 7 | 3 |
| 59 | 44 | 8 | 3 |
| 60 | 45 | 9 | 3 |
| 61 | 46 | 10 | 3 |
| 62 | 47 | 11 | 3 |
| 63 | 48 | 12 | 3 |
| 64 | 49 | 13 | 3 |
| 65 | 50 | 14 | 3 |
| 66 | 51 | 15 | 3 |
| 69 | 54 | 944 | 3 |

---

## 6. Etape 1B - finalisation du corpus

### 6.1 Bibliographies internes aux articles inseres

Le debut est un paragraphe reduit au seul mot « References » ; la fin est la
prochaine entree de la table des matieres. Les citations au fil des phrases
scientifiques ne sont jamais touchees.

| Debut | Fin | Chapitre | Article | Page coupee |
|---|---|---|---|---|
| p64 (par. 7) | p66 | 3 | First-author article published in Weather and Climate Dy | oui |
| p80 (par. 3) | p82 | 3 | Co-author article published in Weather and Climate Dynam | oui |
| p110 (par. 0) | p118 | 4 | First-author article under revision in Earth System Dyna | non |

**15 pages** sorties du corpus RAG (13 entieres, 2 partielles), soit **68,913 caracteres**, conservees dans `backend/data/processed/article_references.jsonl`.

Sur les deux pages coupees, la partie scientifique reste indexee :

- **p64** conserve 1411 caracteres :

```
Code and data availability. The ERA5 reanalysis data are publicly available at https://doi.org/10.24381/cds.143582cf (Copernicus Climate Change Service, 2023). The Ana-SWG code, the processed temperature time series and the analogue files are available on Zenodo at https://doi.or
```

- **p80** conserve 407 caracteres :

```
and Copernicus Climate Change Service (2023) for making available ERA5 data.

Financial support. This research has been supported by the European Commission Horizon 2020 Framework Programme (grant no. 101003469). Sebastian Sippel acknowledges support from the Swiss Data Science C
```

### 6.2 Annexe C - legendes reconstituees

33 pages inspectees, **31 conservees**, 2 exclues. Chaque legende est reconstruite a partir
de son marqueur (« Fig. Sxx. ») ; les labels de panneaux qui la precedent
(« ssp585 ») et les fragments de moins de 12 mots sont ecartes.

**p176** (40 mots) :

```
Fig. S1. Absolute values (contours, in m) and anomalies (shaded areas, in m) with respect to 1950-1999 of 500-hPa geopotential height (Z500) for the 10% coldest SWG simulations (i.e. 100 trajectories) for each period (columns) and SSP (rows) in CNRM-ESM2-1.
```

**p186** (39 mots) :

```
Fig. S11. Standardized standard deviation (shaded areas, σ) and anomalies with respect to 1950-1999 standard deviation of 500-hPa geopotential height (Z500) for the 10% coldest SWG simulations (i.e. 100 trajectories) for each period (columns) and SSP (rows) in CNRM-ESM2-1.
```

**p196** (106 mots) :

```
Fig. S21. Temperature (TG15d) distribution of 1000 SWG simulations for four SSPs (a-d) and three climate periods (left to right) depending on the variable used for importance sampling (colours) in CNRM-ESM2-1. Temperatures are adjusted by the median DJF temperature bias. In the box plots, the boxes represent the median (q50), with the low
```

Pages exclues : p174 (annexe_c_sans_texte_exploitable), p206 (page_blanche).

### 6.3 Mots coupes entre deux pages

Ces reparations ne sont **pas** appliquees au corpus : la provenance page par
page est preservee. Elles seront appliquees par le chunker quand un chunk
traversera la frontiere. Voir `page_boundary_repairs.json`.

| De | Vers | Fragment gauche | Fragment droit | Mot reconstitue | Confiance | A relire |
|---|---|---|---|---|---|---|
| p53 | p54 | `ensem-` | `ble` | **ensemble** | 1.0 | non |
| p57 | p58 | `config-` | `uration` | **configuration** | 1.0 | non |
| p59 | p60 | `win-` | `ter` | **winter** | 1.0 | non |
| p69 | p70 | `cold-` | `wave` | **cold-wave** | 0.75 | oui |
| p71 | p73 | `re-` | `placement` | **replacement** | 0.75 | oui |
| p75 | p76 | `win-` | `ter` | **winter** | 1.0 | non |
| p147 | p148 | `computa-` | `tionally` | **computationally** | 1.0 | non |

La decision vient des frequences reelles du corpus. Quand aucune des deux
formes n'est attestee, on tranche sur la nature des fragments : « cold » et
« wave » existent isolement, c'est donc un mot compose (`cold-wave`) ;
« placement » n'existe jamais seul, c'est donc une coupure syllabique
(`replacement`). Ces deux cas sont signales pour relecture humaine.
