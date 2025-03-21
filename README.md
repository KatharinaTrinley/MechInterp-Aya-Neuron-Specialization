# MechInterp-Aya-Neuron-Specialization

### Overlapping Languages Between Aya101 and EC40

The following table shows the 35 languages that appear in both the Aya101 and EC40 datasets, along with their resourcedness levels in each dataset.

I'll add a column to indicate which languages from the table are included in Aya-23 based on the information you provided.

| ISO Code | Language Name  | Aya101 Resourcedness | EC40 Resourcedness | In Aya-23 |
|----------|---------------|----------------------|-------------------|-----------|
| afr      | Afrikaans     | Mid                  | Low               | No        |
| amh      | Amharic       | Low                  | Low               | No        |
| ara      | Arabic        | High                 | High              | Yes       |
| bel      | Belarusian    | Mid                  | Extremely-Low     | No        |
| ben      | Bengali       | Mid                  | High              | No        |
| bul      | Bulgarian     | Mid                  | Medium            | No        |
| cat      | Catalan       | High                 | Extremely-Low     | No        |
| ces      | Czech         | High                 | High              | Yes       |
| dan      | Danish        | Mid                  | Medium            | No        |
| deu      | German        | High                 | High              | Yes       |
| fra      | French        | High                 | High              | Yes       |
| guj      | Gujarati      | Low                  | Low               | No        |
| hau      | Hausa         | Low                  | Medium            | No        |
| heb      | Hebrew        | Mid                  | High              | Yes       |
| hin      | Hindi         | High                 | High              | Yes       |
| isl      | Icelandic     | Low                  | Extremely-Low     | No        |
| ita      | Italian       | High                 | Medium            | Yes       |
| kan      | Kannada       | Low                  | Medium            | No        |
| ltz      | Luxembourgish | Low                  | Low               | No        |
| mar      | Marathi       | Low                  | Medium            | No        |
| mlt      | Maltese       | Low                  | Medium            | No        |
| nep      | Nepali        | Low                  | Extremely-Low     | No        |
| nld      | Dutch         | High                 | High              | Yes       |
| nor      | Norwegian     | Low                  | Extremely-Low     | No        |
| pol      | Polish        | High                 | Medium            | Yes       |
| por      | Portuguese    | High                 | Medium            | Yes       |
| ron      | Romanian      | Mid                  | Low               | Yes       |
| rus      | Russian       | High                 | High              | Yes       |
| snd      | Sindhi        | Low                  | Low               | No        |
| som      | Somali        | Low                  | Extremely-Low     | No        |
| spa      | Spanish       | High                 | High              | Yes       |
| srp      | Serbian       | High                 | Low               | No        |
| swe      | Swedish       | High                 | Medium            | No        |
| ukr      | Ukrainian     | Mid                  | Low               | Yes       |
| urd      | Urdu          | Mid                  | Extremely-Low     | No        |


Aya-23 also supports these languages that aren't in the EC40 dataset:
- Chinese (simplified & traditional)
- Greek
- Indonesian
- Japanese
- Korean
- Persian
- Turkish
- Vietnamese

## Languages in EC40 but not in Aya101

The analysis identified 5 languages that appear in the EC40 dataset but are not included in the Aya101 dataset:

| Code | ISO Code | Resource Level | Size |
|------|----------|---------------|------|
| oc   | oci      | Low           | 100k |
| ti   | tir      | Low           | 100k |
| ast  | ast      | Extremely-Low | 50k  |
| bs   | bos      | Extremely-Low | 50k  |
| kab  | kab      | Extremely-Low | 50k  |

### Language Families:
| Family       | Languages                                                                                                                                                            |
|-------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Germanic    | German, Dutch, Swedish, Danish, Afrikaans, Luxembourgish, Norwegian, Icelandic, English, Frisian, Faroese, Yiddish, Scots                                            |
| Romance     | French, Spanish, Italian, Portuguese, Romanian, Occitan, Asturian, Catalan, Galician, Corsican, Sicilian, Venetian, Aragonese                                        |
| Slavic      | Russian, Czech, Polish, Bulgarian, Ukrainian, Serbian, Belarusian, Bosnian, Slovak, Slovene, Macedonian, Montenegrin                                                 |
| Indo-Aryan  | Hindi, Bengali, Kannada, Marathi, Sindhi, Gujarati, Nepali, Urdu, Punjabi, Assamese, Sinhala, Konkani, Maithili, Rajasthani, Bhojpuri, Odia                         |
| Afro-Asiatic | Arabic, Hebrew, Maltese, Amharic, Tigrinya, Hausa, Kabyle, Somali, Berber                                                                                          |

---
### EC40

| Resource Level  | Languages                                        | Size  |
|----------------|--------------------------------------------------|------|
| High          | de, nl, fr, es, ru, cs, hi, bn, ar, he           | 5M   |
| Medium        | sv, da, it, pt, pl, bg, kn, mr, mt, ha           | 1M   |
| Low           | af, lb, ro, oc, uk, sr, sd, gu, ti, am           | 100k |
| Extremely-Low | no, is, ast, ca, be, bs, ne, ur, kab, so         | 50k  |



