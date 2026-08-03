# Synthetic Patient Context: Distribution Parameters & Citations

## Overview

IVF-Bench uses 752 real embryo cases from the Kromp Blastocyst Dataset
(Kromp et al., 2023; DOI: 10.1038/s41597-023-02182-3). Each case has **real**
clinical data (age, AMH, endometrial thickness, COCs, MII oocytes, transfer day,
and pregnancy outcomes). For fields that Kromp does not provide, we generate
**synthetic** patient context from published population distributions.

This document records the source, rationale, and specific parameters for every
synthetic field. Fields flagged as synthetic in each case JSON via
`patient_context.synthetic_fields`.

---

## Real Fields (from Kromp clinical_dataset.csv)

| Field | Kromp Column | Notes |
|-------|-------------|-------|
| Age | `Age` | Integer, range 20-45 in dataset |
| AMH (ng/mL) | `AMH` | Float; 165/754 missing; Excel date corruption fixed |
| Endometrial thickness (mm) | `Endo` | Float; 12/754 missing; Excel date corruption fixed |
| COCs retrieved | `COC` | Integer; 5/754 missing; some have "X+Y" format (summed) |
| MII oocytes | `MII` | Integer; 1 "12+5" value (summed to 17) |
| Transfer day | `d` | 4 or 5; all embryos are Day 5 blastocysts |
| Biochemical pregnancy | `SS` | Binary; positive hCG test |
| Clinical pregnancy | `HA` | Binary; clinical heart activity on ultrasound |
| Live birth | `LB` | Binary; cascade validated: LB→HA→SS |
| Gardner grades | `EXP_silver`, `ICM_silver`, `TE_silver` | From clinical CSV directly |

### Data Quality Notes

- **Excel date corruption**: The Kromp CSV was exported from German-locale Excel,
  which converted decimal values like `1.26` into date strings like `Jän.26`
  (January 26). We reverse this corruption using German month abbreviation
  mapping (Jän→1, Feb→2, Mär→3, Apr→4, Mai→5, Jun→6, Jul→7, Aug→8, Sep→9,
  Okt→10, Nov→11, Dez→12). 500 AMH values and 144 Endo values were recovered.
- **Outcome cascade violation**: 1 record (`480_01.png`) has LB=1 but HA=0.
  This is retained as-is (real data anomaly, possibly a data entry error in
  the original dataset).
- **Missing image**: `187_1.png` is listed in the clinical CSV but absent from
  the image directory. This record is excluded (753 usable records).

---

## Synthetic Fields

### 1. Body Mass Index (BMI)

**Distribution**: Truncated normal
**Parameters**: mean = 25.0, SD = 5.0, range [17.0, 45.0]
**Unit**: kg/m²

**Rationale**: IVF patient BMI varies by geography. US NHANES data for infertile
women shows mean 32.57 ± 0.73 kg/m² (Huang et al., 2026). However, IVF-treated
populations skew lower than the general infertile population due to clinic
selection. A large IVF hyperresponder cohort reports BMI 25.1–25.7 kg/m²
(Walls et al., 2024). European IVF populations typically report mean BMI 24–26
(ESHRE registries). We use mean = 25.0 as a representative central estimate.

**Citations**:
- Huang et al. (2026). "Association Between METS-IR and Risk of Infertility."
  *Int J Women's Health*. DOI: 10.2147/IJWH.S575111. PMID: PMC12954206.
  [NHANES n=6,844; infertile women BMI 32.57 ± 0.73]
- Walls et al. (2024). "Clinical outcomes from ART in predicted hyperresponders."
  *Hum Reprod*. DOI: 10.1093/humrep/dead273. PMID: 38177084.
  [n=1,707; BMI 25.1–25.7 for IVF patients]
- Murugappan et al. (2025). "The reproductive journey of women with obesity
  undergoing ART." *Fertil Steril*. DOI: 10.1016/j.fertnstert.2025.07.022.
  PMID: 40685107. [n=48,595 IVF cycles in 31,829 women]

---

### 2. Follicle-Stimulating Hormone (FSH)

**Distribution**: Age-dependent truncated normal
**Parameters**: base mean = 6.0 IU/L at age 30, increases 0.15 IU/L per year
above 30; SD = 1.5; range [1.0, 25.0]
**Unit**: IU/L (basal, day 2-3)

**Rationale**: Basal FSH is a standard ovarian reserve marker. Normal range is
3–10 mIU/mL, with values >10–12 considered elevated. FSH increases with age as
ovarian reserve declines. A study of 1,058 stimulation cycles found that FSH
above the 90th percentile in women ≥35 years carries OR 8.64 for unexpected
poor response (Jovanovic et al., 2024).

**Citations**:
- Jovanovic et al. (2024). "Association of 'normal' early follicular FSH
  concentrations with unexpected poor or suboptimal response." *Reprod Biomed
  Online*. DOI: 10.1016/j.rbmo.2023.103701. PMID: 38309124.
  [n=1,058 cycles; >90th percentile FSH: OR 8.64 for poor response]

**Limitation**: No single large-cohort publication provides mean ± SD of basal
FSH by age for IVF populations. Parameters are based on clinical consensus
ranges (3–10 mIU/mL normal, increasing with age).

---

### 3. Infertility Diagnosis

**Distribution**: Categorical
**Parameters**:

| Diagnosis | Probability | Source |
|-----------|------------|--------|
| Tubal factor | 0.25 | ESHRE registry approximation |
| Male factor | 0.25 | ESHRE registry approximation |
| Unexplained | 0.15 | ESHRE registry; Datta et al. (2024) |
| Endometriosis | 0.10 | ESHRE registry approximation |
| PCOS | 0.12 | ESHRE registry approximation |
| Diminished ovarian reserve | 0.05 | Clinical prevalence |
| Uterine factor | 0.03 | Clinical prevalence |
| Other | 0.05 | Remainder |

**Rationale**: Diagnosis distribution varies substantially by geography and
clinical setting. A Chinese IVF cohort (Xu et al., 2026; n=1,183) reports
tubal factor at 53–59%, reflecting higher tubal disease prevalence in East Asia.
Western registries (ESHRE, SART) show more balanced distributions. We use
approximate ESHRE proportions for a Western-representative benchmark.

**Citations**:
- Xu et al. (2026). "Influencing Factors and Predictive Algorithm of Pregnancy
  Outcomes in IVF/ICSI-ET." *Int J Women's Health*. DOI: 10.2147/IJWH.S577483.
  PMID: PMC12967481. [n=1,183; tubal 53–59%, male 12–14%, unexplained 12–14%]
- Datta et al. (2024). "The definition of unexplained infertility: A systematic
  review." *BJOG*. DOI: 10.1111/1471-0528.17697. PMID: 37957032.
- ESHRE (2025). "ART in Europe, 2020." *Hum Reprod*. DOI: 10.1093/humrep/deaf179.
  PMID: 40985526. [1,440 clinics, 41 countries]

**Limitation**: We use approximate ESHRE proportions rather than exact registry
figures. The diagnosis categories are simplified; real patients often have
multiple contributing factors.

---

### 4. Stimulation Protocol

**Distribution**: Categorical
**Parameters**:

| Protocol | Probability | Source |
|----------|------------|--------|
| GnRH antagonist | 0.70 | Korean nationwide registry |
| Long GnRH agonist | 0.20 | Clinical practice data |
| Short GnRH agonist | 0.05 | Clinical practice data |
| Natural cycle | 0.03 | Clinical practice data |
| Other | 0.02 | Including PPOS, mild stimulation |

**Rationale**: The GnRH antagonist protocol dominates contemporary IVF practice.
A Korean nationwide study (n=10,684) reports 76.8% antagonist protocol usage
(Yoo et al., 2026). European practice shows similar trends with antagonist
protocols comprising 60–80% of cycles. We use 70% as a conservative estimate.

**Citations**:
- Yoo et al. (2026). "Real-World Effectiveness and Safety of rhFSH in Infertile
  Women undergoing ART: A Korean Nationwide Cohort Study." *Clin Pharmacol Ther*.
  DOI: 10.1002/cpt.70136. PMID: 41328812.
  [n=10,684; GnRH antagonist 76.8%]

---

### 5. Previous IVF Cycles

**Distribution**: Categorical
**Parameters**:

| Previous cycles | Probability |
|----------------|------------|
| 0 | 0.40 |
| 1 | 0.25 |
| 2 | 0.15 |
| 3 | 0.10 |
| 4 | 0.05 |
| 5+ | 0.05 |

**Previous cycle outcomes** are sampled from: no pregnancy (45%), biochemical
pregnancy (20%), clinical miscarriage (15%), live birth (15%), ectopic (5%).

**Rationale**: Cumulative live birth rates increase with successive cycles. A
large Belgian cohort (n=31,478 embryos, 11,463 women) reports cumulative LBR
of 51.1% after 3 transfers, 68.3% after 6, and 78.0% after 10 (Vander Borght
et al., 2025). Second-cycle LBR by age: <35: 42.3%, 35–37: 42.7%, 38–40:
25.5%, >40: 16.2% (Liu et al., 2024). The distribution approximates a
geometric decay reflecting that most patients achieve pregnancy within 1–3
cycles.

**Citations**:
- Vander Borght et al. (2025). "Cumulative live birth rates of 31,478 untested
  embryos challenge traditional RIF definitions." *Hum Reprod*. DOI:
  10.1093/humrep/deaf036. PMID: 40064027. [cLBR: 3rd=51.1%, 6th=68.3%, 10th=78%]
- Liu et al. (2024). "Embryo development and live birth in women with one
  previously failed IVF cycle." *J Assist Reprod Genet*. DOI:
  10.1007/s10815-024-03107-8. PMID: 38739214.
  [2nd cycle LBR by age: <35: 42.3%, >40: 16.2%]

---

### 6. Sperm Parameters

**Distribution**: Truncated normal
**Parameters**:
- Concentration: mean = 50.0 M/mL, SD = 25.0, range [5.0, 200.0]
- Motility: mean = 55.0%, SD = 15.0, range [10.0, 95.0]

**Rationale**: The WHO Laboratory Manual for Semen Analysis (6th edition, 2021)
defines lower reference limits (5th percentile): concentration ≥15 M/mL, total
motility ≥40%, progressive motility ≥32%. Population means are substantially
higher than these lower limits. A large PGT-A cohort (n=3,101 couples) applies
WHO-2021 criteria showing that below-5th-percentile values reduce euploid
blastocyst rates by 2.7–5.8% (Beukers et al., 2025).

**Citations**:
- WHO (2021). *WHO Laboratory Manual for the Examination and Processing of
  Human Semen*, 6th edition.
- Beukers et al. (2025). "A WHO 2021-based scheme outlining sperm parameters'
  associations with IVF outcomes in PGT-A cycles." *Andrology*. DOI:
  10.1111/andr.13811. PMID: 39609097. [n=3,101 couples, 4,013 ICSI cycles]
- Stankovic et al. (2026). "Diagnostic accuracy of sperm DNA fragmentation
  index in male infertility." *J Med Biochem*. DOI: 10.5937/jomb0-59605.
  PMID: PMC12967198. [WHO reference limits: conc ≥15M/mL, motility ≥40%]

---

### 7. Partner (Male) Age

**Distribution**: Correlated with female age
**Parameters**: offset = Normal(mean=2.4, SD=4.3) added to female age;
clipped to range [20, 65]

**Rationale**: In IVF couples, male partners are typically 2–3 years older than
female partners. A Czech infertility clinic cohort (n=469 couples) reports women
mean age 30.79 ± 4.65, men mean age 33.19 ± 4.94, with mean difference 2.4
years (SD=4.3) and inter-partner age correlation r=0.60 (Jelinkova et al., 2026).
A Chinese IVF cohort (n=1,183) reports female median 32.0, male median 33.0
(Xu et al., 2026). Paternal age ≥45 does not significantly reduce LBR (aOR 0.93,
n=56,113 FET cycles; Du et al., 2024).

**Citations**:
- Jelinkova et al. (2026). "Sociodemographic and medical correlates of
  fertility-related QoL in primary infertile Czech couples." *Health Psychol
  Rep*. DOI: 10.5114/hpr/213967. PMID: PMC12968782.
  [Women 30.79±4.65, Men 33.19±4.94, difference 2.4±4.3yr, r=0.60]
- Xu et al. (2026). *Int J Women's Health*. DOI: 10.2147/IJWH.S577483.
  [Female median 32, male median 33]
- Du et al. (2024). "Paternal age does not jeopardize LBR after IVF." *Am J
  Obstet Gynecol*. DOI: 10.1016/j.ajog.2023.11.1224. PMID: 37952870.
  [n=56,113 FET; age ≥45 aOR 0.93 (0.79–1.10)]

---

### 8. Lifestyle Profile

The lifestyle field is composed from **five independently sampled dimensions**,
each derived from published IVF/ART patient cohort data. Dimensions are sampled
independently per case and composed into a semicolon-separated string (e.g.,
"never smoker; light drinker (1-3/wk); moderate exercise (1-3x/wk);
health-conscious diet; mild stress").

#### 8a. Smoking Status

**Distribution**: Categorical

| Status | Probability | Source |
|--------|------------|--------|
| Never smoker | 0.65 | Gaskins 73%, Romanian 54%, weighted mean |
| Former smoker | 0.25 | Romanian 26%, Joelsson ~25% |
| Current smoker | 0.10 | Dodge 2–7%, Joelsson 3–10%, Romanian 20% |

**Citations**:
- Dodge et al. (2015). "Lifestyle habits of 12,800 IVF patients." *Hum Fertil*
  18(4):253-257. DOI: 10.3109/14647273.2015.1071881. [n=12,811; current 2–7%]
- Joelsson et al. (2019). "Effect of lifestyle risk factors on oocytes in IVF."
  *PLOS ONE* 14(8):e0221015. DOI: 10.1371/journal.pone.0221015.
  [n=673; current 3.3–9.5%, any smoking 32–35%]
- Gaskins et al. (2019). *Am J Obstet Gynecol* 220(6):567.e1-18.
  DOI: 10.1016/j.ajog.2019.02.004. [n=357 EARTH; never smoked 73%]
- Romanian pilot (2022). *PeerJ* 10:e14189. DOI: 10.7717/peerj.14189.
  [n=35; current 20%, former 26%, never 54%]

#### 8b. Alcohol Consumption

**Distribution**: Categorical

| Category | Probability | Source |
|----------|------------|--------|
| None or rare (<1/wk) | 0.40 | Dodge 77–83% abstain during tx; Rossi ~78% <50 g/wk |
| Light (1–3 drinks/wk) | 0.35 | Joelsson 69–78% any consumption; composite estimate |
| Moderate (4–6 drinks/wk) | 0.20 | Rossi ~22% ≥50 g/wk (split light/moderate) |
| Heavy (7+ drinks/wk) | 0.05 | Tail estimate from Rossi high-intake group |

**Citations**:
- Rossi et al. (2011). "Effect of Alcohol Consumption on IVF." *Obstet
  Gynecol* 117(1):136-142. DOI: 10.1097/AOG.0b013e31820090e1.
  [n=2,545 couples, 4,729 cycles; total alcohol <50 g/wk (n=3,572 cycles)
  vs ≥50 g/wk (n=1,034 cycles); per-beverage: <1 vs 1–7 drinks/wk for
  beer, wine, liquor separately]
- Dodge et al. (2015). [n=12,811; 17–23% any alcohol during treatment]
- Joelsson et al. (2019). [n=673; 69–78% any alcohol consumption]
- Lyngsø et al. (2021). "Female cigarette smoking and successful fertility
  treatment." *Acta Obstet Gynecol Scand* 100(1):49-58.
  DOI: 10.1111/aogs.13979. [n=1,708 Danish cohort]

#### 8c. Physical Activity / Exercise

**Distribution**: Categorical

| Level | Probability | Source |
|-------|------------|--------|
| Sedentary | 0.30 | Sherwin 29% low IPAQ, Romanian 46% never |
| Light (walking only) | 0.25 | Sherwin: 72% no structured exercise |
| Moderate (1–3x/wk) | 0.25 | Sherwin 46% moderate IPAQ |
| Active (4+x/wk) | 0.15 | Sherwin 25% high IPAQ |
| Very active (daily) | 0.05 | Tail estimate |

**Citations**:
- Sherwin et al. (2022). "Habitual physical activity levels in women attending
  infertility clinic." *Reprod Fertil* 3(3):178-186.
  DOI: 10.1530/RAF-22-0067. [n=229 IPAQ-SF; low 29%, moderate 46%, high 25%;
  72% no structured exercise; mean sitting 5±4 hr/day]
- Dodge et al. (2015). [47–62% exercised during IVF treatment]
- Romanian pilot (2022). [never exercise 46%, 1–2x/wk 14%, 3–4x/wk 17%,
  daily 17%]

#### 8d. Dietary Pattern

**Distribution**: Categorical

| Pattern | Probability | Source |
|---------|------------|--------|
| Standard Western | 0.50 | Default; Gaskins Q1-Q2 ~54% |
| Health-conscious | 0.25 | Gaskins profertility Q3-Q4 ~46% |
| Mediterranean | 0.15 | Karayiannis 2018 study population |
| Vegetarian/vegan | 0.05 | General population proxy (~5%) |
| Restricted | 0.05 | ~5% of reproductive-age women |

**Citations**:
- Gaskins et al. (2019). "Dietary Patterns and Outcomes of Assisted
  Reproduction." *Am J Obstet Gynecol* 220(6):567.e1-18.
  DOI: 10.1016/j.ajog.2019.02.004.
  [n=357, 608 cycles; profertility diet Q1 29%, Q2 25%, Q3 22%, Q4 24%;
  multivitamin use 69–99%]
- Karayiannis et al. (2018). "Adherence to the Mediterranean diet and IVF
  success rate." *Hum Reprod* 33(3):494-502.
  DOI: 10.1093/humrep/dex376. [n=244 non-obese women; higher adherence →
  65–68% greater likelihood of clinical pregnancy in women <35]

#### 8e. Stress Level

**Distribution**: Categorical

| Level | Probability | Source |
|-------|------------|--------|
| Low | 0.15 | Below population norms |
| Mild | 0.30 | STAI-S 35–39 range |
| Moderate | 0.35 | Mean STAI-S ~41 (elevated over norm 35) |
| High | 0.15 | STAI-S 50–60 range |
| Severe/clinical | 0.05 | Meets clinical threshold |

**Citations**:
- Holley et al. (2015). "Prospective Study of Depression and Anxiety in
  Female Fertility Preservation and Infertility Patients." *Fertil Steril*.
  PMC4253550. [pre-treatment: depression 14%, anxiety (STAI-S ≥39) 27%;
  post-treatment: depression 23%, anxiety 51%]
- Klonoff-Cohen & Natarajan (2004). "Stress and Anxiety in IVF Cycles."
  [STAI-S mean 41.45±13.09 at baseline, elevated over female norm 35.20]

**Limitations**: Stress is self-reported and subject to measurement
variability. The STAI-S thresholds used for binning are approximate. Diet
pattern categories are simplified from continuous dietary quality scores.
Vegetarian/vegan prevalence among IVF patients is estimated from general
population data. All five dimensions are sampled independently — real-world
correlations (e.g., smokers may exercise less) are not modeled.

---

## Reproducibility

All synthetic values are deterministically generated via per-case SHA-256
seeding: `seed = SHA256(base_seed + ":" + case_id)[:8]` converted to a 32-bit
integer. This ensures:
- Adding or removing cases does not change other cases' profiles
- Regenerating any case always produces identical values
- No dependency on generation order

Base seed for patient context: `123` (configurable via YAML).

---

## References (consolidated)

1. Kromp F, et al. (2023). "An annotated human blastocyst dataset to benchmark
   deep learning architectures for in vitro fertilization." *Sci Data* 10:271.
   DOI: 10.1038/s41597-023-02182-3.
2. ESHRE (2025). "ART in Europe, 2020." *Hum Reprod*. DOI: 10.1093/humrep/deaf179.
3. Huang et al. (2026). *Int J Women's Health*. DOI: 10.2147/IJWH.S575111.
4. Walls et al. (2024). *Hum Reprod*. DOI: 10.1093/humrep/dead273.
5. Murugappan et al. (2025). *Fertil Steril*. DOI: 10.1016/j.fertnstert.2025.07.022.
6. Jovanovic et al. (2024). *Reprod Biomed Online*. DOI: 10.1016/j.rbmo.2023.103701.
7. Xu et al. (2026). *Int J Women's Health*. DOI: 10.2147/IJWH.S577483.
8. Datta et al. (2024). *BJOG*. DOI: 10.1111/1471-0528.17697.
9. Yoo et al. (2026). *Clin Pharmacol Ther*. DOI: 10.1002/cpt.70136.
10. Vander Borght et al. (2025). *Hum Reprod*. DOI: 10.1093/humrep/deaf036.
11. Liu et al. (2024). *J Assist Reprod Genet*. DOI: 10.1007/s10815-024-03107-8.
12. WHO (2021). *Laboratory Manual for Semen Analysis*, 6th edition.
13. Beukers et al. (2025). *Andrology*. DOI: 10.1111/andr.13811.
14. Stankovic et al. (2026). *J Med Biochem*. DOI: 10.5937/jomb0-59605.
15. Jelinkova et al. (2026). *Health Psychol Rep*. DOI: 10.5114/hpr/213967.
16. Du et al. (2024). *Am J Obstet Gynecol*. DOI: 10.1016/j.ajog.2023.11.1224.
17. Frontiers in Endocrinology (2025). AMH nomogram, n=22,920. DOI: 10.3389/fendo.2025.1612194.
18. SART (2025). EMT-outcome analysis, n=182,784. DOI: 10.1016/j.fertnstert.2025.04.032.
19. Dodge et al. (2015). "Lifestyle habits of 12,800 IVF patients." *Hum Fertil*
    18(4):253-257. DOI: 10.3109/14647273.2015.1071881.
20. Joelsson et al. (2019). "Effect of lifestyle risk factors on oocytes in IVF."
    *PLOS ONE* 14(8):e0221015. DOI: 10.1371/journal.pone.0221015.
21. Rossi et al. (2011). "Effect of Alcohol Consumption on IVF." *Obstet Gynecol*
    117(1):136-142. DOI: 10.1097/AOG.0b013e31820090e1. PMC4487775.
    [n=2,545 couples, 4,729 cycles; alcohol in g/wk and per-beverage categories]
22. Lyngsø et al. (2021). "Female cigarette smoking and successful fertility
    treatment." *Acta Obstet Gynecol Scand* 100(1):49-58. DOI: 10.1111/aogs.13979.
23. Sherwin et al. (2022). "Habitual physical activity levels in women attending
    infertility clinic." *Reprod Fertil* 3(3):178-186. DOI: 10.1530/RAF-22-0067.
    PMC9578060.
24. Gaskins et al. (2019). "Dietary Patterns and Outcomes of Assisted Reproduction."
    *Am J Obstet Gynecol* 220(6):567.e1-18. DOI: 10.1016/j.ajog.2019.02.004.
    PMC6545142.
25. Karayiannis et al. (2018). "Adherence to the Mediterranean diet and IVF
    success rate." *Hum Reprod* 33(3):494-502. DOI: 10.1093/humrep/dex376.
26. Holley et al. (2015). "Prospective Study of Depression and Anxiety in Female
    Fertility Preservation and Infertility Patients." *Fertil Steril*. PMC4253550.
27. Romanian pilot (2022). "Specific lifestyle factors and IVF outcomes in
    Romanian women." *PeerJ* 10:e14189. DOI: 10.7717/peerj.14189.
28. Homan et al. (2007). "Impact of lifestyle factors on reproductive performance."
    *Hum Reprod Update* 13(3):209-223. DOI: 10.1093/humupd/dml056.
