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

##RMSE for the 25M dataset
<img
  src="/docs/metrics_with_featues_25m_page-0001.jpg"
  alt="Alt text"
  title="Optional title"
  style="display: inline-block; margin: 0 auto; max-width: 300px">
Predictions
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
