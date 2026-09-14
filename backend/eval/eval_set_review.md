# Jeu d'evaluation - revue des 24 questions

Version 1.0.0, genere le 2026-08-01T08:14:43+00:00.

Chaque page et chaque extrait ci-dessous a ete verifie automatiquement dans
`backend/data/processed/corpus_clean.jsonl` : le script de construction echoue si
une preuve est introuvable. Les questions sont en francais, le corpus est a 94 %
en anglais : c'est le cas d'usage cross-lingue vise par le projet.

## Tableau de synthese

| ID | Type | Diff. | Split | Question | Route | Pages PDF | Pages these | Entites | Relation | Statut |
|---|---|---|---|---|---|---|---|---|---|---|
| `SEM_001` | semantic | easy | train | Comment l'Organisation météorologique mondiale définit-elle une vague de froid | `use_vector` | [19] | [4] | World Meteorological Organisation, cold wave | — | verified |
| `SEM_002` | semantic | easy | train | Qu'est-ce que l'indice WCC utilisé dans la thèse et comment est-il calculé ? | `use_vector` | [96] | [81] | WCC, Z500 | — | verified |
| `SEM_003` | semantic | medium | train | Qu'est-ce que l'ajustement dynamique (dynamical adjustment) et à quoi sert-il  | `use_vector` | [70] | [55] | dynamical adjustment, atmospheric circulation | — | verified |
| `SEM_004` | semantic | medium | test | Pourquoi la thèse utilise-t-elle un générateur stochastique de temps avec écha | `use_vector` | [30] | [15] | SWG, empirical importance sampling, analogues of circulation | — | verified |
| `SEM_005` | semantic | hard | train | Quel était l'objectif général de la thèse et pourquoi les événements de froid  | `use_vector` | [122] | [107] | climate change, extreme cold events, midlatitudes | — | verified |
| `SEM_006` | semantic | hard | test | Comment la thèse justifie-t-elle la valeur retenue pour le paramètre alpha_T d | `use_vector` | [45] | [30] | SWG, winter 1963 | — | verified |
| `REL_001` | relational | easy | train | Par quel trajet le blocage scandinave amène-t-il l'air froid, et quelle en est | `use_graph` | [23] | [8] | Scandinavian Blocking, Siberia, Western Europe, France | Scandinavian Blocking -[CAUSES]-> vagues de froid les plus i | verified |
| `REL_002` | relational | easy | train | Quels sont les quatre régimes de temps qui caractérisent la météo hivernale en | `use_graph` | [21] | [6] | NAO+, NAO-, Atlantic ridge, Scandinavian Blocking | NAO+, NAO-, Atlantic ridge, Scandinavian Blocking -[CHARACTE | verified |
| `REL_003` | relational | medium | train | À quelle phase de l'Oscillation nord-atlantique l'hiver 1963 est-il associé, e | `use_graph` | [53] | [38] | winter 1963, NAO, Iceland low, Azores high | winter 1963 -[ASSOCIATED_WITH]-> phase negative de la NAO | verified |
| `REL_004` | relational | medium | test | Quelle configuration du courant-jet Harnik et ses coauteurs relient-ils à la p | `use_graph` | [21] | [6] | merged jet, NAO | merged jet configuration -[LINKED_TO]-> phase negative de la | verified |
| `REL_005` | relational | hard | train | Quel effet l'amplification arctique pourrait-elle avoir sur la météo hivernale | `use_graph` | [53] | [38] | Arctic Amplification, midlatitudes, severe winter weather | Arctic Amplification -[MAY_INCREASE]-> conditions hivernales | verified |
| `REL_006` | relational | hard | test | Quel événement stratosphérique de janvier 1963 a pu contribuer à maintenir les | `use_graph` | [69] | [54] | sudden stratospheric warming, polar vortex, Europe | sudden stratospheric warming de janvier 1963 -[WEAKENED_AND_ | verified |
| `HYB_001` | hybrid | easy | train | Quel lien existe entre le blocage scandinave et l'intensité des vagues de froi | `use_hybrid` | [22, 23] | [7, 8] | Scandinavian Blocking, NAO, Western Europe, Greenland | Scandinavian Blocking et NAO negative -[SHARE_MECHANISM]-> b | verified |
| `HYB_002` | hybrid | easy | train | Quelle relation la thèse établit-elle entre l'hiver 1963 et la NAO, et comment | `use_hybrid` | [53, 69] | [38, 54] | winter 1963, NAO, jet stream | winter 1963 -[ASSOCIATED_WITH]-> NAO negative et courant-jet | verified |
| `HYB_003` | hybrid | medium | train | À quoi la thèse utilise-t-elle l'indice WCC vis-à-vis des modèles CMIP6, et po | `use_hybrid` | [87, 96] | [72, 81] | WCC, CMIP6, NAO, France | WCC -[OUTPERFORMS_FOR_COLD_SPELLS]-> indice NAO quotidien cl | verified |
| `HYB_004` | hybrid | medium | test | Quel est le lien entre le générateur stochastique de temps et les analogues de | `use_hybrid` | [30, 31] | [15, 16] | SWG, analogues of circulation, extreme cold events | SWG -[BASED_ON]-> analogues de circulation | verified |
| `HYB_005` | hybrid | hard | train | Quelle relation la thèse établit-elle entre le SWG et la méthode d'ensemble bo | `use_hybrid` | [95, 122] | [80, 107] | SWG, ensemble boosting, Gessner et al. | SWG -[COMPARED_TO]-> ensemble boosting de Gessner et al. (20 | verified |
| `HYB_006` | hybrid | hard | test | Quelle contrainte le nombre d'analogues impose-t-il aux résultats du SWG, et q | `use_hybrid` | [123] | [108] | SWG, analogues, CMIP6 | nombre d'analogues K -[CONSTRAINS]-> resultats du SWG | verified |
| `OOC_001` | out_of_context | easy | train | Quelles ont été les causes de la « rain bomb » australienne de 2022 ? | `abstain` | — | — | — | — | verified |
| `OOC_002` | out_of_context | medium | train | Comment des ensembles de simulations climatiques ont-ils servi à anticiper les | `abstain` | — | — | — | — | verified |
| `OOC_003` | out_of_context | hard | test | Quelle méthode d'attribution au changement climatique a été appliquée aux préc | `abstain` | — | — | — | — | verified |
| `OOC_004` | out_of_context | easy | train | Quels sont les principaux symptômes de la grippe saisonnière ? | `abstain` | — | — | — | — | verified |
| `OOC_005` | out_of_context | medium | train | Comment fonctionne l'algorithme de rétropropagation du gradient dans un réseau | `abstain` | — | — | — | — | verified |
| `OOC_006` | out_of_context | hard | test | Quelles sont les conséquences de l'inflation sur le marché immobilier français | `abstain` | — | — | — | — | verified |

## Reponses attendues et preuves

### SEM_001 — semantic — easy — train

**Question.** Comment l'Organisation météorologique mondiale définit-elle une vague de froid ?

**Route attendue.** `use_vector`

**Reponse attendue.** L'Organisation météorologique mondiale (WMO) définit une vague de froid comme un événement météorologique généralement caractérisé par une chute brutale de la température de l'air près de la surface terrestre, conduisant à des valeurs extrêmement basses pouvant s'accompagner de conditions dangereuses telles que le gel et le verglas.

- **Page PDF 19** (page these 4, langue `en`) — chapitre 1, section 1.1.2 « Definition of extreme cold events »

  > In its guidelines the World Meteorological Organisation (WMO) defines a cold wave as "a meteorological event generally characterized by a sharp drop in air temperature near the Earth's surface, leading to extremely low values that can be associated with hazardous weather, such as frost and icing" (WMO 2023).

_Note : Definition officielle citee mot pour mot dans la these._

### SEM_002 — semantic — easy — train

**Question.** Qu'est-ce que l'indice WCC utilisé dans la thèse et comment est-il calculé ?

**Route attendue.** `use_vector`

**Reponse attendue.** Le WCC (Western Europe Cold Circulation index) est un indice qui caractérise le schéma de circulation atmosphérique associé aux vagues de froid en Europe de l'Ouest. Il se calcule en soustrayant la moyenne du géopotentiel Z500 entre les zones (1°W - 9°E ; 40°N - 47°N) et (24°W - 13°W ; 62°N - 66°N), zones choisies comme maximum et minimum de la structure dipolaire identifiée.

- **Page PDF 96** (page these 81, langue `en`) — chapitre 4, section 3.4 « SWG with importance sampling on circulation »

  > Therefore, to investigate the dynamics of cold spells in France, we compute a Western Europe Cold Circulation index (WCC), which characterizes this atmospheric pattern, by subtracting the mean of Z500 between (1°W - 9°E; 40°N - 47°N) and (24°W - 13°W; 62°N - 66°N).

- **Page PDF 96** (page these 81, langue `en`) — chapitre 4, section 3.4 « SWG with importance sampling on circulation »

  > Those areas were chosen as the maximum and minimum of the dipole structure identified from the Z500 composite map of the 20 coldest cold spells in France from ERA5.

_Note : Definition et formule donnees dans la section methodes du chapitre 4._

### SEM_003 — semantic — medium — train

**Question.** Qu'est-ce que l'ajustement dynamique (dynamical adjustment) et à quoi sert-il ?

**Route attendue.** `use_vector`

**Reponse attendue.** L'ajustement dynamique est une technique des sciences du climat qui vise à estimer l'influence de la circulation atmosphérique sur une variable climatique de surface, par exemple la température de l'air en surface. Dans la thèse, il sert à séparer la composante induite par la circulation de la composante résiduelle (thermodynamique) des tendances de température hivernale.

- **Page PDF 70** (page these 55, langue `en`) — chapitre 3, section 2 « Methods and data »

  > Dynamical adjustment is a technique in climate science, which aims to estimate the influence of atmospheric circulation on a target surface climate variable, such as surface air temperature (Wallace et al., 1995; Smoliak et al., 2015; Deser et al., 2016).

- **Page PDF 70** (page these 55, langue `en`) — chapitre 3, section 2 « Methods and data »

  > 1 by characterizing circulation-induced and residual (thermodynamical) trends of the domain-averaged winter temperature time series over the Germany domain.

_Note : Definition explicite en section 2.2 de l'article du chapitre 3._

### SEM_004 — semantic — medium — test

**Question.** Pourquoi la thèse utilise-t-elle un générateur stochastique de temps avec échantillonnage d'importance empirique ?

**Route attendue.** `use_vector`

**Reponse attendue.** Parce qu'il s'agit d'une méthode peu coûteuse en calcul qui combine des approches à la fois statistiques et physiques. Elle repose sur une discrétisation de l'espace des phases par les analogues de circulation, et elle est particulièrement adaptée aux événements de moyenne à longue durée, c'est-à-dire durant plus d'une semaine.

- **Page PDF 30** (page these 15, langue `en`) — chapitre 1, section None « Research questions »

  > In this thesis, I use an SWG with empirical importance sampling, which is a computationally efficient method that combines both statistical and physical approaches.

- **Page PDF 30** (page these 15, langue `en`) — chapitre 1, section None « Research questions »

  > This method is particularly well-suited for mid- to long-term events (i.e., lasting more than a week) and thus fits well the purpose of this work.

_Note : Justification methodologique donnee en section 1.2 du chapitre 1._

### SEM_005 — semantic — hard — train

**Question.** Quel était l'objectif général de la thèse et pourquoi les événements de froid extrême restent-ils peu étudiés ?

**Route attendue.** `use_vector`

**Reponse attendue.** L'objectif général était d'explorer comment le changement climatique affecte les événements de froid extrême aux moyennes latitudes, et plus particulièrement en Europe. Ces événements restent peu étudiés parce que l'attention se concentre sur les vagues de chaleur extrême ; leurs impacts pourraient donc être sous-estimés, alors qu'ils affectent encore fortement la santé et les systèmes énergétiques.

- **Page PDF 122** (page these 107, langue `en`) — chapitre 5, section 5.1 « Conclusions »

  > 111 5.1 Conclusions The overall objective of this thesis was to explore how climate change impacts extreme cold events in the midlatitudes, and more specifically Europe.

- **Page PDF 122** (page these 107, langue `en`) — chapitre 5, section 5.1 « Conclusions »

  > Cold extremes remain understudied because of the focus on extreme heat events and their subsequent impacts could be underestimated.

- **Page PDF 122** (page these 107, langue `en`) — chapitre 5, section 5.1 « Conclusions »

  > Light has indeed been shed on heatwaves with global warming but, despite their decrease in intensity, cold events are still causing significant impacts on health and energy systems.

_Note : Chapitre 5, section Conclusions._

### SEM_006 — semantic — hard — test

**Question.** Comment la thèse justifie-t-elle la valeur retenue pour le paramètre alpha_T du générateur stochastique ?

**Route attendue.** `use_vector`

**Reponse attendue.** Avec une valeur de 0,2, seules quelques simulations atteignent des températures plus froides que l'hiver 1963. Une valeur supérieure à 0,5 permet de simuler une plus grande proportion d'événements extrêmes, mais les différences deviennent moins significatives au-delà de ce seuil. L'auteure a donc retenu la valeur 0,5 pour la suite de l'analyse.

- **Page PDF 45** (page these 30, langue `en`) — chapitre 2, section 2.3.3 « Parameter adjustment of the SWG »

  > With αT = 0.2, only a few simulations reach temperatures colder than winter 1963.

- **Page PDF 45** (page these 30, langue `en`) — chapitre 2, section 2.3.3 « Parameter adjustment of the SWG »

  > A value of αT exceeding 0.5 enables the simulation of a higher proportion of extreme events.

- **Page PDF 45** (page these 30, langue `en`) — chapitre 2, section 2.3.3 « Parameter adjustment of the SWG »

  > However, since the differences become less significant beyond this threshold, I selected αT = 0.5 for the subsequent analysis.

_Note : Test de sensibilite du parametre, chapitre 2 section 2.3.3._

### REL_001 — relational — easy — train

**Question.** Par quel trajet le blocage scandinave amène-t-il l'air froid, et quelle en est la conséquence pour la France ?

**Route attendue.** `use_graph`

**Reponse attendue.** Un blocage scandinave amène l'air froid en passant par la Sibérie et le nord-est de l'Europe. Cet air reste au-dessus des surfaces continentales ; cette configuration, parfois appelée « Moscou-Paris express », provoque généralement les vagues de froid les plus intenses sur l'Europe de l'Ouest et la France.

- **Page PDF 23** (page these 8, langue `en`) — chapitre 1, section 1.1.4 « Observed evolution of extreme cold events »

  > A Scandinavian blocking brings cold air through Siberia and the North-East of Europe.

- **Page PDF 23** (page these 8, langue `en`) — chapitre 1, section 1.1.4 « Observed evolution of extreme cold events »

  > Therefore this configuration, sometimes referred to as Moscow-Paris express, usually causes the most intense cold spells over Western Europe and France (Fig.

**Relation attendue.** `Scandinavian Blocking` -[CAUSES]-> `vagues de froid les plus intenses en Europe de l'Ouest et en France`

_Note : Relation causale explicite, chapitre 1 section 1.1.3._

### REL_002 — relational — easy — train

**Question.** Quels sont les quatre régimes de temps qui caractérisent la météo hivernale en Atlantique Nord selon la thèse ?

**Route attendue.** `use_graph`

**Reponse attendue.** Les quatre régimes sont la phase positive et la phase négative de l'Oscillation nord-atlantique (NAO+ et NAO−), qui sont opposées, la dorsale atlantique (Atlantic ridge, AR) et le blocage scandinave (Scandinavian blocking, SB).

- **Page PDF 21** (page these 6, langue `en`) — chapitre 1, section 1.1.3 « Dynamics of cold spells in Western Europe »

  > Four main regimes are usually sufficient to characterize wintertime weather: the NAO+ and NAO-, opposite phases of the NAO, the Atlantic ridge (AR) and the Scandinavian blocking (SB) (Madonna et al.

**Relation attendue.** `NAO+, NAO-, Atlantic ridge, Scandinavian Blocking` -[CHARACTERIZE]-> `la circulation hivernale en Atlantique Nord`

_Note : Enumeration des regimes de temps, chapitre 1 section 1.1.3._

### REL_003 — relational — medium — train

**Question.** À quelle phase de l'Oscillation nord-atlantique l'hiver 1963 est-il associé, et que traduit cet indice ?

**Route attendue.** `use_graph`

**Reponse attendue.** L'hiver 1963 est associé à un indice NAO négatif. Un indice négatif traduit une différence de pression plus faible que la normale entre la dépression d'Islande et l'anticyclone des Açores.

- **Page PDF 53** (page these 38, langue `en`) — chapitre 3, section 1 « Introduction »

  > Winter 1963 was associated with a negative North Atlantic Oscillation (NAO) index, indicating a lower-than-normal pressure difference between the Iceland low- and Azores high-pressure systems (Cattiaux et al., 2010; Greatbatch et al., 2015).

**Relation attendue.** `winter 1963` -[ASSOCIATED_WITH]-> `phase negative de la NAO`

_Note : Chapitre 3, introduction du premier article._

### REL_004 — relational — medium — test

**Question.** Quelle configuration du courant-jet Harnik et ses coauteurs relient-ils à la phase négative de la NAO ?

**Route attendue.** `use_graph`

**Reponse attendue.** Harnik et al. (2014) relient la configuration de jet fusionné (merged jet) à une phase négative de l'Oscillation nord-atlantique et à des hivers anormaux.

- **Page PDF 21** (page these 6, langue `en`) — chapitre 1, section 1.1.3 « Dynamics of cold spells in Western Europe »

  > (2014) link the merged jet configuration to a negative phase of the North Atlantic Oscillation (NAO) and anomalous winters.

**Relation attendue.** `merged jet configuration` -[LINKED_TO]-> `phase negative de la NAO`

_Note : Relation attribuee a une reference precise, chapitre 1 section 1.1.3._

### REL_005 — relational — hard — train

**Question.** Quel effet l'amplification arctique pourrait-elle avoir sur la météo hivernale des moyennes latitudes ?

**Route attendue.** `use_graph`

**Reponse attendue.** L'amplification arctique est présentée comme un mécanisme susceptible de conduire à une augmentation des conditions hivernales sévères aux moyennes latitudes. La thèse souligne toutefois que cet effet potentiel reste débattu et entouré d'incertitudes.

- **Page PDF 53** (page these 38, langue `en`) — chapitre 3, section 1 « Introduction »

  > Arctic amplification (AA) is a mechanism that may lead to an increase in severe winter weather in the midlatitudes (Cohen et al., 2014; Francis et al., 2018; Francis and Vavrus, 2012; Vavrus, 2018), but its potential effect is intertwined with other hemispheric drivers of decadal variability, and the quantification of its influence remains debated (Blackport and Screen, 2020; Cohen et al., 2020; Francis, 2017).

**Relation attendue.** `Arctic Amplification` -[MAY_INCREASE]-> `conditions hivernales severes aux moyennes latitudes`

_Note : Relation explicitement presentee comme incertaine et debattue._

### REL_006 — relational — hard — test

**Question.** Quel événement stratosphérique de janvier 1963 a pu contribuer à maintenir les conditions anormales en Europe, et par quel mécanisme ?

**Route attendue.** `use_graph`

**Reponse attendue.** Un réchauffement stratosphérique soudain (SSW) s'est produit fin janvier 1963. Il s'est accompagné d'un affaiblissement du vortex polaire, ce qui a pu contribuer à maintenir la persistance des conditions anormales en Europe tout au long du mois de février.

- **Page PDF 69** (page these 54, langue `en`) — chapitre 3, section 1 « Introduction »

  > In late January 1963, a sudden stratospheric warming event took place with the associated weakened polar vortex and may have helped to maintain persistence in the anomalous conditions throughout February in Europe (Greatbatch et al., 2015).

**Relation attendue.** `sudden stratospheric warming de janvier 1963` -[WEAKENED_AND_MAINTAINED]-> `vortex polaire affaibli et persistance des conditions anormales en Europe`

_Note : Formulation prudente dans la these (« may have helped »)._

### HYB_001 — hybrid — easy — train

**Question.** Quel lien existe entre le blocage scandinave et l'intensité des vagues de froid en Europe de l'Ouest, et qu'ont en commun ce régime et la NAO négative ?

**Route attendue.** `use_hybrid`

**Reponse attendue.** Le blocage scandinave amène l'air froid par la Sibérie et le nord-est de l'Europe ; cet air restant au-dessus des terres, cette configuration provoque les vagues de froid les plus intenses sur l'Europe de l'Ouest et la France. Ce régime et la phase négative de la NAO ont en commun un blocage sur l'Atlantique Nord, situé soit sur le Groenland, soit sur la Scandinavie.

- **Page PDF 23** (page these 8, langue `en`) — chapitre 1, section 1.1.4 « Observed evolution of extreme cold events »

  > A Scandinavian blocking brings cold air through Siberia and the North-East of Europe.

- **Page PDF 23** (page these 8, langue `en`) — chapitre 1, section 1.1.4 « Observed evolution of extreme cold events »

  > That air remains over land areas.

- **Page PDF 22** (page these 7, langue `en`) — chapitre 1, section 1.1.3 « Dynamics of cold spells in Western Europe »

  > Both these patterns have in common a blocking over the North-Atlantic, either over Greenland or Scandinavia.

**Relation attendue.** `Scandinavian Blocking et NAO negative` -[SHARE_MECHANISM]-> `blocage sur l'Atlantique Nord (Groenland ou Scandinavie)`

_Note : Necessite deux pages : la relation causale (p23) et le mecanisme commun (p22)._

### HYB_002 — hybrid — easy — train

**Question.** Quelle relation la thèse établit-elle entre l'hiver 1963 et la NAO, et comment le courant-jet explique-t-il ces conditions ?

**Route attendue.** `use_hybrid`

**Reponse attendue.** L'hiver 1963 est associé à un indice NAO négatif, traduisant une différence de pression plus faible que la normale entre la dépression d'Islande et l'anticyclone des Açores. Du point de vue dynamique, le courant-jet et les vents d'ouest associés ont été scindés et déplacés très au sud, la dépression étant décalée à l'est des Açores.

- **Page PDF 53** (page these 38, langue `en`) — chapitre 3, section 1 « Introduction »

  > Winter 1963 was associated with a negative North Atlantic Oscillation (NAO) index, indicating a lower-than-normal pressure difference between the Iceland low- and Azores high-pressure systems (Cattiaux et al., 2010; Greatbatch et al., 2015).

- **Page PDF 69** (page these 54, langue `en`) — chapitre 3, section 1 « Introduction »

  > While the NAO was not extremely negative, as the low was displaced to the east of the Azores (Cadiou and Yiou, 2024), the jet stream and associated westerlies were split and displaced far to the north and south of their usual position (O'Connor, 1963).

**Relation attendue.** `winter 1963` -[ASSOCIATED_WITH]-> `NAO negative et courant-jet scinde et deplace vers le sud`

_Note : Relation (p53) + explication dynamique dans un second article (p69)._

### HYB_003 — hybrid — medium — train

**Question.** À quoi la thèse utilise-t-elle l'indice WCC vis-à-vis des modèles CMIP6, et pourquoi cet indice est-il préférable à un indice NAO classique ?

**Route attendue.** `use_hybrid`

**Reponse attendue.** L'indice WCC sert d'abord à évaluer dans quelle mesure les modèles CMIP6 reproduisent des mécanismes semblables à ceux observés sur la période historique, puis à établir un lien causal entre la circulation atmosphérique identifiée et les vagues de froid extrême en France. Il est préférable à un indice NAO quotidien classique parce qu'il est corrélé plus fortement à la température quotidienne en France en hiver (r = 0,65 contre r = 0,36 pour la NAO), étant taillé pour ce type d'événement et cette région.

- **Page PDF 87** (page these 72, langue `en`) — chapitre 4, section 4.2 « First-author article under revision in Earth »

  > I used this index first to assess how well CMIP6 models reproduce similar mechanisms to those observed over the historical period, and secondly to establish a causal link between the identified atmospheric circulation and extreme cold spells in France across climates with increasing levels of warming.

- **Page PDF 96** (page these 81, langue `en`) — chapitre 4, section 3.4 « SWG with importance sampling on circulation »

  > 4, the WCC is correlated to daily temperature over France during winter months (Pearson correlation coefficient r = 0.65 and p < 10-15).

- **Page PDF 96** (page these 81, langue `en`) — chapitre 4, section 3.4 « SWG with importance sampling on circulation »

  > As this WCC index is tailored for a specific type of event (15-day cold spells) over a specific region (metropolitan France), it performs better than a classic daily North Atlantic Oscillation (NAO) index, defined as the normalized SLP difference between the Azores and Iceland (r = 0.36 and p < 10-15).

**Relation attendue.** `WCC` -[OUTPERFORMS_FOR_COLD_SPELLS]-> `indice NAO quotidien classique`

_Note : Deux pages et deux chapitres differents : usage (p87) et performance (p96)._

### HYB_004 — hybrid — medium — test

**Question.** Quel est le lien entre le générateur stochastique de temps et les analogues de circulation, et quel objectif de la thèse cela sert-il ?

**Route attendue.** `use_hybrid`

**Reponse attendue.** Le générateur stochastique de temps repose sur une discrétisation de l'espace des phases par les analogues de circulation. Ce lien sert le premier objectif de la thèse : adapter et améliorer le SWG existant, fondé sur ces analogues, pour simuler des événements de froid extrême dans une approche de type storyline.

- **Page PDF 30** (page these 15, langue `en`) — chapitre 1, section None « Research questions »

  > It comprises a discretization of the phase-space by analogues of circulation.

- **Page PDF 31** (page these 16, langue `en`) — chapitre 1, section 1.3 « Research questions »

  > Therefore, the first objective of this thesis is to adapt and improve the existing SWG based on analogues of circulation to the simulation of extreme cold events for a storyline approach.

**Relation attendue.** `SWG` -[BASED_ON]-> `analogues de circulation`

_Note : Relation methodologique (p30) rattachee a l'objectif de recherche (p31)._

### HYB_005 — hybrid — hard — train

**Question.** Quelle relation la thèse établit-elle entre le SWG et la méthode d'ensemble boosting, et comment explique-t-elle l'écart d'intensité observé ?

**Route attendue.** `use_hybrid`

**Reponse attendue.** La thèse compare le SWG à un autre algorithme d'événements rares, la méthode de splitting dite « ensemble boosting » de Gessner et al. (2021) ; l'approche du SWG, qui consiste à rejouer des trajectoires alternatives d'événements extrêmes préexistants, en est proche. Les deux méthodes donnent des résultats très similaires et cohérents. Les hivers simulés par le SWG sont toutefois légèrement moins intenses, parce que le SWG est contraint par ses données d'entrée.

- **Page PDF 122** (page these 107, langue `en`) — chapitre 5, section 5.1 « Conclusions »

  > I compared the SWG results to those of another rare event algorithm: the splitting method of ensemble boosting by Gessner et al. (2021).

- **Page PDF 122** (page these 107, langue `en`) — chapitre 5, section 5.1 « Conclusions »

  > The SWG and ensemble boosting yielded very similar and coherent results, despite the two algorithms using different approaches, thus confirming the validity of the implementation of SWG for extreme cold events.

- **Page PDF 122** (page these 107, langue `en`) — chapitre 5, section 5.1 « Conclusions »

  > The winters simulated by the SWG were slightly less intense than those with ensemble boosting, due to the SWG's constraint by input data.

- **Page PDF 95** (page these 80, langue `en`) — chapitre 4, section 3.2 « Protocol of SWG simulations »

  > This SWG approach of running alternative trajectories of pre-existing extreme events is similar to the "ensemble boosting" method of Gessner et al. (2021).

**Relation attendue.** `SWG` -[COMPARED_TO]-> `ensemble boosting de Gessner et al. (2021)`

_Note : Relation entre methodes (p122) et similarite d'approche (p95)._

### HYB_006 — hybrid — hard — test

**Question.** Quelle contrainte le nombre d'analogues impose-t-il aux résultats du SWG, et quelle amélioration la thèse propose-t-elle ?

**Route attendue.** `use_hybrid`

**Reponse attendue.** Se limiter à K = 20 analogues, pour préserver la qualité des analogues, contraint les résultats du SWG. La thèse propose de calculer les analogues sur les membres d'ensembles de simulations de grande taille, tels que ceux fournis par certains modèles CMIP6 : cela permettrait d'utiliser un nombre d'analogues plus élevé à chaque pas de temps tout en maintenant une qualité d'analogues élevée.

- **Page PDF 123** (page these 108, langue `en`) — chapitre 5, section 5.2 « Caveats and Perspectives »

  > However, being limited to K = 20 analogues for the sake of analogue quality does constrain the SWG results.

- **Page PDF 123** (page these 108, langue `en`) — chapitre 5, section 5.2 « Caveats and Perspectives »

  > Further studies could be improved by computing analogues over ensemble members of large ensemble simulations, as provided by some CMIP6 models.

- **Page PDF 123** (page these 108, langue `en`) — chapitre 5, section 5.2 « Caveats and Perspectives »

  > This would enable running the SWG with a higher number of analogues (K) at each time step while maintaining high analogue quality.

**Relation attendue.** `nombre d'analogues K` -[CONSTRAINS]-> `resultats du SWG`

_Note : Limite (p123) et perspective d'amelioration (p123), chapitre 5._

### OOC_001 — out_of_context — easy — train

**Question.** Quelles ont été les causes de la « rain bomb » australienne de 2022 ?

**Route attendue.** `abstain`

**Reponse attendue.** Je ne sais pas.

**Origine hors contexte.** annexe B, exclue du corpus RAG principal.

Aucune page, aucune preuve : le systeme doit s'abstenir.

_Note : Sujet de l'annexe B, volontairement exclue du corpus RAG principal._

### OOC_002 — out_of_context — medium — train

**Question.** Comment des ensembles de simulations climatiques ont-ils servi à anticiper les canicules des Jeux olympiques de Paris 2024 ?

**Route attendue.** `abstain`

**Reponse attendue.** Je ne sais pas.

**Origine hors contexte.** annexe B, exclue du corpus RAG principal.

Aucune page, aucune preuve : le systeme doit s'abstenir.

_Note : Second article de l'annexe B, hors du perimetre de la these._

### OOC_003 — out_of_context — hard — test

**Question.** Quelle méthode d'attribution au changement climatique a été appliquée aux précipitations extrêmes du Queensland en 2022 ?

**Route attendue.** `abstain`

**Reponse attendue.** Je ne sais pas.

**Origine hors contexte.** annexe B, exclue du corpus RAG principal.

Aucune page, aucune preuve : le systeme doit s'abstenir.

_Note : Annexe B. Le texte existe dans annex_b_out_of_context.jsonl mais ce fichier n'appartient pas au corpus RAG : la reponse attendue reste « Je ne sais pas. »_

### OOC_004 — out_of_context — easy — train

**Question.** Quels sont les principaux symptômes de la grippe saisonnière ?

**Route attendue.** `abstain`

**Reponse attendue.** Je ne sais pas.

**Origine hors contexte.** sujet totalement etranger a la these.

Aucune page, aucune preuve : le systeme doit s'abstenir.

_Note : Sujet medical, totalement etranger au corpus._

### OOC_005 — out_of_context — medium — train

**Question.** Comment fonctionne l'algorithme de rétropropagation du gradient dans un réseau de neurones ?

**Route attendue.** `abstain`

**Reponse attendue.** Je ne sais pas.

**Origine hors contexte.** sujet totalement etranger a la these.

Aucune page, aucune preuve : le systeme doit s'abstenir.

_Note : Sujet d'apprentissage automatique, absent de la these._

### OOC_006 — out_of_context — hard — test

**Question.** Quelles sont les conséquences de l'inflation sur le marché immobilier français ?

**Route attendue.** `abstain`

**Reponse attendue.** Je ne sais pas.

**Origine hors contexte.** sujet totalement etranger a la these.

Aucune page, aucune preuve : le systeme doit s'abstenir.

_Note : Sujet economique, absent de la these. Piege : contient « francais », present dans le corpus, mais aucun contenu ne soutient la reponse._
