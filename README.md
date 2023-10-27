# Recommender System Using ALS Model For Colaborative Filtering

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![AGPL License](https://img.shields.io/badge/license-AGPL-blue.svg)](http://www.gnu.org/licenses/agpl-3.0)


The project is aims to create a Collaborative recommenders ystem to predict rating for movie lense dataset. The underlying technique of colaborative filter depend is Matrix factorization and ALS as a optimizer. It is the type of technique used for to work on sparse data.




## Usage

Download the files from this repo then runs the code below

```bash
  python3 train.py --path 'data/ratings_small.csv' --epoch=10 --ts 0.1
```
There are different parameters to change
```bash
  python3 train.py --path 'data/ratings_small.csv' --epoch=10 --ts 0.1 lamda 0.01 thau 0.01
```
Given the dummy use we can also predict the movies for the 25 million dataset
```bash
  python predict.py --path parameters/parameters_25m.pickle --n 0 --fact 1
```

## RMSE for the 25M dataset
<img
  src="/docs/metrics_with_featues_25m_page-0001.jpg"
  alt="Alt text"
  title="Optional title"
  style="display: inline-block; max-width: 200px;height:300px">
  
## Predictions
prediction for dummy user 1 and 2
```bash
                                                   title                                        genres
192                                         Smoke (1995)                                  Comedy|Drama
335                                      War, The (1994)                           Adventure|Drama|War
351                                  Forrest Gump (1994)                      Comedy|Drama|Romance|War
362                                     Mask, The (1994)                   Action|Comedy|Crime|Fantasy
429                                   Cliffhanger (1993)                     Action|Adventure|Thriller
437                                Demolition Man (1993)                       Action|Adventure|Sci-Fi
452                                 Fugitive, The (1993)                                      Thriller
536                                  Blade Runner (1982)                        Action|Sci-Fi|Thriller
585                     Silence of the Lambs, The (1991)                         Crime|Horror|Thriller
650                     James and the Giant Peach (1996)  Adventure|Animation|Children|Fantasy|Musical
934                              Bringing Up Baby (1938)                                Comedy|Romance
948                            African Queen, The (1951)                  Adventure|Comedy|Romance
------------------------------------------------------------------------------------------------------
                                                  title                                     genres
46                           Seven (a.k.a. Se7en) (1995)                           Mystery|Thriller
80        Things to Do in Denver When You're Dead (1995)                        Crime|Drama|Romance
219                             Circle of Friends (1995)                              Drama|Romance
231                                  Exit to Eden (1994)                                     Comedy
259                            Little Princess, A (1995)                             Children|Drama
429                                   Cliffhanger (1993)                  Action|Adventure|Thriller
475                                 Jurassic Park (1993)           Action|Adventure|Sci-Fi|Thriller
545               Nightmare Before Christmas, The (1993)         Animation|Children|Fantasy|Musical
628                                Jack and Sarah (1995)                                    Romance
642                                   Dragonheart (1996)                   Action|Adventure|Fantasy
734    Dr. Strangelove or: How I Learned to Stop Worr...                                 Comedy|War
840                                Godfather, The (1972)                                Crime|Drama
899                            Gone with the Wind (1939)                          Drama|Romance|War
944                                 39 Steps, The (1935)                     Drama|Mystery|Thriller
1035         William Shakespeare's Romeo + Juliet (1996)                              Drama|Romance
1083   Microcosmos (Microcosmos: Le peuple de l'herbe...                                Documentary
1120         Wallace & Gromit: The Wrong Trousers (1993)            Animation|Children|Comedy|Crime
1229                            Great Escape, The (1963)                 Action|Adventure|Drama|War
1251                               Big Sleep, The (1946)                    Crime|Film-Noir|Mystery
1371                                      Michael (1996)               Comedy|Drama|Fantasy|Romance
1527                                      Contact (1997)                               Drama|Sci-Fi
