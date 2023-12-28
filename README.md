# Recommender System Using Colaborative Filtering

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![AGPL License](https://img.shields.io/badge/license-AGPL-blue.svg)](http://www.gnu.org/licenses/agpl-3.0)


The project is aims to create a Collaborative recommenders system to predict rating for movie lense dataset. 
The underlying technique of colaborative filter depend is Matrix factorization and ALS as a optimizer.
It is the type of technique used for to work on sparse data.




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

## Prediction
prediciton for user who give lord of the rings 5 star rating. User 99.
```bash
                                                 title                                           genres
461                          Hot Shots! Part Deux (1993)                                Action|Comedy|War
522                              Schindler's List (1993)                                        Drama|War
585                     Silence of the Lambs, The (1991)                            Crime|Horror|Thriller
710                         Great White Hype, The (1996)                                           Comedy
926                                My Man Godfrey (1936)                                   Comedy|Romance
1170                                       Aliens (1986)                   Action|Adventure|Horror|Sci-Fi
1182                                   Goodfellas (1990)                                      Crime|Drama
1186                                          Ran (1985)                                        Drama|War
1249                                     Fantasia (1940)               Animation|Children|Fantasy|Musical
1252                                     Heathers (1989)                                           Comedy
1262                          Room with a View, A (1986)                                    Drama|Romance
1483            Shall We Dance? (Shall We Dansu?) (1996)                             Comedy|Drama|Romance
4572                               Turner & Hooch (1989)                                     Comedy|Crime
4772                                 Donnie Darko (2001)                    Drama|Mystery|Sci-Fi|Thriller
4887   Lord of the Rings: The Fellowship of the Ring,...                                Adventure|Fantasy
4889                            Beautiful Mind, A (2001)                                    Drama|Romance
7039                                    Peter Pan (2003)                Action|Adventure|Children|Fantasy
7052                                        Osama (2003)                                            Drama
8269                               Ocean's Twelve (2004)                     Action|Comedy|Crime|Thriller
10253                                      Capote (2005)                                      Crime|Drama
10838                                   Clerks II (2006)                                           Comedy
11163                               Prestige, The (2006)                    Drama|Mystery|Sci-Fi|Thriller
11435                                      Zodiac (2007)                             Crime|Drama|Thriller
11700   Harry Potter and the Order of the Phoenix (2007)                     Adventure|Drama|Fantasy|IMAX
11862                            30 Days of Night (2007)                                  Horror|Thriller
13285                                   Star Trek (2009)                     Action|Adventure|Sci-Fi|IMAX
13531        Imaginarium of Doctor Parnassus, The (2009)                                    Drama|Fantasy
13575                                  District 9 (2009)                          Mystery|Sci-Fi|Thriller
14156                                 Daybreakers (2010)                     Action|Drama|Horror|Thriller
14983                                  Mr. Nobody (2009)                     Drama|Fantasy|Romance|Sci-Fi
16523                             Kung Fu Panda 2 (2011)  Action|Adventure|Animation|Children|Comedy|IMAX
17674                                Intouchables (2011)                                     Comedy|Drama
19928                            Lone Ranger, The (2013)                    Action|Adventure|Western|IMAX
25070                   Guardians of the Galaxy 2 (2017)                          Action|Adventure|Sci-Fi
26834  A Pigeon Sat on a Branch Reflecting on Existen...                                     Comedy|Drama
37182                         10 Cloverfield Lane (2016)                                         Thriller
51294                  All the Money in the World (2017)                     Crime|Drama|Mystery|Thriller


