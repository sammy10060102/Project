{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "from matplotlib.pyplot import rcParams\n",
    "import seaborn as sns\n",
    "import numpy as np\n",
    "from scipy.stats import norm\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,StandardScaler,MinMaxScaler,PowerTransformer,FunctionTransformer\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.tree import DecisionTreeRegressor\n",
    "from sklearn.model_selection  import train_test_split\n",
    "from sklearn.metrics import mean_squared_error\n",
    "from scipy import stats\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>age</th>\n",
       "      <th>sex</th>\n",
       "      <th>bmi</th>\n",
       "      <th>children</th>\n",
       "      <th>smoker</th>\n",
       "      <th>region</th>\n",
       "      <th>charges</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>19</td>\n",
       "      <td>female</td>\n",
       "      <td>27.900</td>\n",
       "      <td>0</td>\n",
       "      <td>yes</td>\n",
       "      <td>southwest</td>\n",
       "      <td>16884.92400</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>18</td>\n",
       "      <td>male</td>\n",
       "      <td>33.770</td>\n",
       "      <td>1</td>\n",
       "      <td>no</td>\n",
       "      <td>southeast</td>\n",
       "      <td>1725.55230</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>28</td>\n",
       "      <td>male</td>\n",
       "      <td>33.000</td>\n",
       "      <td>3</td>\n",
       "      <td>no</td>\n",
       "      <td>southeast</td>\n",
       "      <td>4449.46200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>33</td>\n",
       "      <td>male</td>\n",
       "      <td>22.705</td>\n",
       "      <td>0</td>\n",
       "      <td>no</td>\n",
       "      <td>northwest</td>\n",
       "      <td>21984.47061</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>32</td>\n",
       "      <td>male</td>\n",
       "      <td>28.880</td>\n",
       "      <td>0</td>\n",
       "      <td>no</td>\n",
       "      <td>northwest</td>\n",
       "      <td>3866.85520</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1333</th>\n",
       "      <td>50</td>\n",
       "      <td>male</td>\n",
       "      <td>30.970</td>\n",
       "      <td>3</td>\n",
       "      <td>no</td>\n",
       "      <td>northwest</td>\n",
       "      <td>10600.54830</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1334</th>\n",
       "      <td>18</td>\n",
       "      <td>female</td>\n",
       "      <td>31.920</td>\n",
       "      <td>0</td>\n",
       "      <td>no</td>\n",
       "      <td>northeast</td>\n",
       "      <td>2205.98080</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1335</th>\n",
       "      <td>18</td>\n",
       "      <td>female</td>\n",
       "      <td>36.850</td>\n",
       "      <td>0</td>\n",
       "      <td>no</td>\n",
       "      <td>southeast</td>\n",
       "      <td>1629.83350</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1336</th>\n",
       "      <td>21</td>\n",
       "      <td>female</td>\n",
       "      <td>25.800</td>\n",
       "      <td>0</td>\n",
       "      <td>no</td>\n",
       "      <td>southwest</td>\n",
       "      <td>2007.94500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1337</th>\n",
       "      <td>61</td>\n",
       "      <td>female</td>\n",
       "      <td>29.070</td>\n",
       "      <td>0</td>\n",
       "      <td>yes</td>\n",
       "      <td>northwest</td>\n",
       "      <td>29141.36030</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>1338 rows × 7 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "      age     sex     bmi  children smoker     region      charges\n",
       "0      19  female  27.900         0    yes  southwest  16884.92400\n",
       "1      18    male  33.770         1     no  southeast   1725.55230\n",
       "2      28    male  33.000         3     no  southeast   4449.46200\n",
       "3      33    male  22.705         0     no  northwest  21984.47061\n",
       "4      32    male  28.880         0     no  northwest   3866.85520\n",
       "...   ...     ...     ...       ...    ...        ...          ...\n",
       "1333   50    male  30.970         3     no  northwest  10600.54830\n",
       "1334   18  female  31.920         0     no  northeast   2205.98080\n",
       "1335   18  female  36.850         0     no  southeast   1629.83350\n",
       "1336   21  female  25.800         0     no  southwest   2007.94500\n",
       "1337   61  female  29.070         0    yes  northwest  29141.36030\n",
       "\n",
       "[1338 rows x 7 columns]"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#read data\n",
    "\n",
    "df = pd.read_csv('./insurance_data.csv')\n",
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 1338 entries, 0 to 1337\n",
      "Data columns (total 7 columns):\n",
      " #   Column    Non-Null Count  Dtype  \n",
      "---  ------    --------------  -----  \n",
      " 0   age       1338 non-null   int64  \n",
      " 1   sex       1338 non-null   object \n",
      " 2   bmi       1338 non-null   float64\n",
      " 3   children  1338 non-null   int64  \n",
      " 4   smoker    1338 non-null   object \n",
      " 5   region    1338 non-null   object \n",
      " 6   charges   1338 non-null   float64\n",
      "dtypes: float64(2), int64(2), object(3)\n",
      "memory usage: 73.3+ KB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Missing_Number</th>\n",
       "      <th>Missing_Percent</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>charges</th>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>region</th>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>smoker</th>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>children</th>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>bmi</th>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>sex</th>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>age</th>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          Missing_Number  Missing_Percent\n",
       "charges                0              0.0\n",
       "region                 0              0.0\n",
       "smoker                 0              0.0\n",
       "children               0              0.0\n",
       "bmi                    0              0.0\n",
       "sex                    0              0.0\n",
       "age                    0              0.0"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#check for missing values \n",
    "def missing_values(df):\n",
    "    null_v = df.isnull().sum().sort_values(ascending=False)\n",
    "    null_percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)\n",
    "    null_v = pd.concat([null_v, null_percent], axis=1, keys=['Missing_Number', 'Missing_Percent'])\n",
    "    return null_v\n",
    "\n",
    "missing_values(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#check for duplicated rows\n",
    "\n",
    "df.duplicated().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>age</th>\n",
       "      <th>sex</th>\n",
       "      <th>bmi</th>\n",
       "      <th>children</th>\n",
       "      <th>smoker</th>\n",
       "      <th>region</th>\n",
       "      <th>charges</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>195</th>\n",
       "      <td>19</td>\n",
       "      <td>male</td>\n",
       "      <td>30.59</td>\n",
       "      <td>0</td>\n",
       "      <td>no</td>\n",
       "      <td>northwest</td>\n",
       "      <td>1639.5631</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>581</th>\n",
       "      <td>19</td>\n",
       "      <td>male</td>\n",
       "      <td>30.59</td>\n",
       "      <td>0</td>\n",
       "      <td>no</td>\n",
       "      <td>northwest</td>\n",
       "      <td>1639.5631</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     age   sex    bmi  children smoker     region    charges\n",
       "195   19  male  30.59         0     no  northwest  1639.5631\n",
       "581   19  male  30.59         0     no  northwest  1639.5631"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#showing duplicated rows\n",
    "df[df.duplicated(keep=False)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "#dropping the duplicated rows \n",
    "df=df.drop_duplicates(keep=\"first\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.duplicated().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>age</th>\n",
       "      <th>bmi</th>\n",
       "      <th>children</th>\n",
       "      <th>charges</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>1337.000000</td>\n",
       "      <td>1337.000000</td>\n",
       "      <td>1337.000000</td>\n",
       "      <td>1337.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>39.222139</td>\n",
       "      <td>30.663452</td>\n",
       "      <td>1.095737</td>\n",
       "      <td>13279.121487</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>14.044333</td>\n",
       "      <td>6.100468</td>\n",
       "      <td>1.205571</td>\n",
       "      <td>12110.359656</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>18.000000</td>\n",
       "      <td>15.960000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1121.873900</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>27.000000</td>\n",
       "      <td>26.290000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>4746.344000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>39.000000</td>\n",
       "      <td>30.400000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>9386.161300</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>51.000000</td>\n",
       "      <td>34.700000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>16657.717450</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>64.000000</td>\n",
       "      <td>53.130000</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>63770.428010</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "               age          bmi     children       charges\n",
       "count  1337.000000  1337.000000  1337.000000   1337.000000\n",
       "mean     39.222139    30.663452     1.095737  13279.121487\n",
       "std      14.044333     6.100468     1.205571  12110.359656\n",
       "min      18.000000    15.960000     0.000000   1121.873900\n",
       "25%      27.000000    26.290000     0.000000   4746.344000\n",
       "50%      39.000000    30.400000     1.000000   9386.161300\n",
       "75%      51.000000    34.700000     2.000000  16657.717450\n",
       "max      64.000000    53.130000     5.000000  63770.428010"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style  type=\"text/css\" >\n",
       "#T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow0_col0,#T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow1_col0,#T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow2_col0,#T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow2_col1,#T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow2_col2,#T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow2_col3,#T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow2_col4,#T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow2_col5,#T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow2_col6,#T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow2_col7,#T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow3_col0{\n",
       "            width:  10em;\n",
       "             height:  80%;\n",
       "        }#T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow0_col1,#T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow0_col6,#T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow1_col5{\n",
       "            width:  10em;\n",
       "             height:  80%;\n",
       "            background:  linear-gradient(90deg,#d65f5f 0.3%, transparent 0.3%);\n",
       "        }#T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow0_col2,#T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow0_col7,#T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow1_col7{\n",
       "            width:  10em;\n",
       "             height:  80%;\n",
       "            background:  linear-gradient(90deg,#d65f5f 0.1%, transparent 0.1%);\n",
       "        }#T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow0_col3{\n",
       "            width:  10em;\n",
       "             height:  80%;\n",
       "            background:  linear-gradient(90deg,#d65f5f 1.6%, transparent 1.6%);\n",
       "        }#T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow0_col4,#T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow1_col4{\n",
       "            width:  10em;\n",
       "             height:  80%;\n",
       "            background:  linear-gradient(90deg,#d65f5f 0.6%, transparent 0.6%);\n",
       "        }#T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow0_col5{\n",
       "            width:  10em;\n",
       "             height:  80%;\n",
       "            background:  linear-gradient(90deg,#d65f5f 0.4%, transparent 0.4%);\n",
       "        }#T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow1_col1,#T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow1_col6{\n",
       "            width:  10em;\n",
       "             height:  80%;\n",
       "            background:  linear-gradient(90deg,#d65f5f 0.2%, transparent 0.2%);\n",
       "        }#T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow1_col2{\n",
       "            width:  10em;\n",
       "             height:  80%;\n",
       "            background:  linear-gradient(90deg,#d65f5f 0.0%, transparent 0.0%);\n",
       "        }#T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow1_col3{\n",
       "            width:  10em;\n",
       "             height:  80%;\n",
       "            background:  linear-gradient(90deg,#d65f5f 1.4%, transparent 1.4%);\n",
       "        }#T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow3_col1,#T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow3_col2,#T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow3_col3,#T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow3_col4,#T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow3_col5,#T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow3_col6,#T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow3_col7{\n",
       "            width:  10em;\n",
       "             height:  80%;\n",
       "            background:  linear-gradient(90deg,#d65f5f 100.0%, transparent 100.0%);\n",
       "        }</style><table id=\"T_8ac2b93a_3000_11ec_90f8_f45c89c1618d\" ><thead>    <tr>        <th class=\"blank level0\" ></th>        <th class=\"col_heading level0 col0\" >count</th>        <th class=\"col_heading level0 col1\" >mean</th>        <th class=\"col_heading level0 col2\" >std</th>        <th class=\"col_heading level0 col3\" >min</th>        <th class=\"col_heading level0 col4\" >25%</th>        <th class=\"col_heading level0 col5\" >50%</th>        <th class=\"col_heading level0 col6\" >75%</th>        <th class=\"col_heading level0 col7\" >max</th>    </tr></thead><tbody>\n",
       "                <tr>\n",
       "                        <th id=\"T_8ac2b93a_3000_11ec_90f8_f45c89c1618dlevel0_row0\" class=\"row_heading level0 row0\" >age</th>\n",
       "                        <td id=\"T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow0_col0\" class=\"data row0 col0\" >1337.000000</td>\n",
       "                        <td id=\"T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow0_col1\" class=\"data row0 col1\" >39.222139</td>\n",
       "                        <td id=\"T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow0_col2\" class=\"data row0 col2\" >14.044333</td>\n",
       "                        <td id=\"T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow0_col3\" class=\"data row0 col3\" >18.000000</td>\n",
       "                        <td id=\"T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow0_col4\" class=\"data row0 col4\" >27.000000</td>\n",
       "                        <td id=\"T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow0_col5\" class=\"data row0 col5\" >39.000000</td>\n",
       "                        <td id=\"T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow0_col6\" class=\"data row0 col6\" >51.000000</td>\n",
       "                        <td id=\"T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow0_col7\" class=\"data row0 col7\" >64.000000</td>\n",
       "            </tr>\n",
       "            <tr>\n",
       "                        <th id=\"T_8ac2b93a_3000_11ec_90f8_f45c89c1618dlevel0_row1\" class=\"row_heading level0 row1\" >bmi</th>\n",
       "                        <td id=\"T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow1_col0\" class=\"data row1 col0\" >1337.000000</td>\n",
       "                        <td id=\"T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow1_col1\" class=\"data row1 col1\" >30.663452</td>\n",
       "                        <td id=\"T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow1_col2\" class=\"data row1 col2\" >6.100468</td>\n",
       "                        <td id=\"T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow1_col3\" class=\"data row1 col3\" >15.960000</td>\n",
       "                        <td id=\"T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow1_col4\" class=\"data row1 col4\" >26.290000</td>\n",
       "                        <td id=\"T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow1_col5\" class=\"data row1 col5\" >30.400000</td>\n",
       "                        <td id=\"T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow1_col6\" class=\"data row1 col6\" >34.700000</td>\n",
       "                        <td id=\"T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow1_col7\" class=\"data row1 col7\" >53.130000</td>\n",
       "            </tr>\n",
       "            <tr>\n",
       "                        <th id=\"T_8ac2b93a_3000_11ec_90f8_f45c89c1618dlevel0_row2\" class=\"row_heading level0 row2\" >children</th>\n",
       "                        <td id=\"T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow2_col0\" class=\"data row2 col0\" >1337.000000</td>\n",
       "                        <td id=\"T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow2_col1\" class=\"data row2 col1\" >1.095737</td>\n",
       "                        <td id=\"T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow2_col2\" class=\"data row2 col2\" >1.205571</td>\n",
       "                        <td id=\"T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow2_col3\" class=\"data row2 col3\" >0.000000</td>\n",
       "                        <td id=\"T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow2_col4\" class=\"data row2 col4\" >0.000000</td>\n",
       "                        <td id=\"T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow2_col5\" class=\"data row2 col5\" >1.000000</td>\n",
       "                        <td id=\"T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow2_col6\" class=\"data row2 col6\" >2.000000</td>\n",
       "                        <td id=\"T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow2_col7\" class=\"data row2 col7\" >5.000000</td>\n",
       "            </tr>\n",
       "            <tr>\n",
       "                        <th id=\"T_8ac2b93a_3000_11ec_90f8_f45c89c1618dlevel0_row3\" class=\"row_heading level0 row3\" >charges</th>\n",
       "                        <td id=\"T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow3_col0\" class=\"data row3 col0\" >1337.000000</td>\n",
       "                        <td id=\"T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow3_col1\" class=\"data row3 col1\" >13279.121487</td>\n",
       "                        <td id=\"T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow3_col2\" class=\"data row3 col2\" >12110.359656</td>\n",
       "                        <td id=\"T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow3_col3\" class=\"data row3 col3\" >1121.873900</td>\n",
       "                        <td id=\"T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow3_col4\" class=\"data row3 col4\" >4746.344000</td>\n",
       "                        <td id=\"T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow3_col5\" class=\"data row3 col5\" >9386.161300</td>\n",
       "                        <td id=\"T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow3_col6\" class=\"data row3 col6\" >16657.717450</td>\n",
       "                        <td id=\"T_8ac2b93a_3000_11ec_90f8_f45c89c1618drow3_col7\" class=\"data row3 col7\" >63770.428010</td>\n",
       "            </tr>\n",
       "    </tbody></table>"
      ],
      "text/plain": [
       "<pandas.io.formats.style.Styler at 0x12209fd60>"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#check data description\n",
    "df.describe().T.style.bar()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>age</th>\n",
       "      <th>sex</th>\n",
       "      <th>bmi</th>\n",
       "      <th>children</th>\n",
       "      <th>smoker</th>\n",
       "      <th>region</th>\n",
       "      <th>charges</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>543</th>\n",
       "      <td>54</td>\n",
       "      <td>female</td>\n",
       "      <td>47.41</td>\n",
       "      <td>0</td>\n",
       "      <td>yes</td>\n",
       "      <td>southeast</td>\n",
       "      <td>63770.42801</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     age     sex    bmi  children smoker     region      charges\n",
       "543   54  female  47.41         0    yes  southeast  63770.42801"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#locating row that has the max price \n",
    "df.loc[df['charges'] == 63770.428010]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>age</th>\n",
       "      <th>sex</th>\n",
       "      <th>bmi</th>\n",
       "      <th>children</th>\n",
       "      <th>smoker</th>\n",
       "      <th>region</th>\n",
       "      <th>charges</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1317</th>\n",
       "      <td>18</td>\n",
       "      <td>male</td>\n",
       "      <td>53.13</td>\n",
       "      <td>0</td>\n",
       "      <td>no</td>\n",
       "      <td>southeast</td>\n",
       "      <td>1163.4627</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      age   sex    bmi  children smoker     region    charges\n",
       "1317   18  male  53.13         0     no  southeast  1163.4627"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#locating row that has the max BMI\n",
    "df.loc[df['bmi'] == 53.130000]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>age</th>\n",
       "      <th>sex</th>\n",
       "      <th>bmi</th>\n",
       "      <th>children</th>\n",
       "      <th>smoker</th>\n",
       "      <th>region</th>\n",
       "      <th>charges</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>32</th>\n",
       "      <td>19</td>\n",
       "      <td>female</td>\n",
       "      <td>28.600</td>\n",
       "      <td>5</td>\n",
       "      <td>no</td>\n",
       "      <td>southwest</td>\n",
       "      <td>4687.79700</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>71</th>\n",
       "      <td>31</td>\n",
       "      <td>male</td>\n",
       "      <td>28.500</td>\n",
       "      <td>5</td>\n",
       "      <td>no</td>\n",
       "      <td>northeast</td>\n",
       "      <td>6799.45800</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>166</th>\n",
       "      <td>20</td>\n",
       "      <td>female</td>\n",
       "      <td>37.000</td>\n",
       "      <td>5</td>\n",
       "      <td>no</td>\n",
       "      <td>southwest</td>\n",
       "      <td>4830.63000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>413</th>\n",
       "      <td>25</td>\n",
       "      <td>male</td>\n",
       "      <td>23.900</td>\n",
       "      <td>5</td>\n",
       "      <td>no</td>\n",
       "      <td>southwest</td>\n",
       "      <td>5080.09600</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>425</th>\n",
       "      <td>45</td>\n",
       "      <td>male</td>\n",
       "      <td>24.310</td>\n",
       "      <td>5</td>\n",
       "      <td>no</td>\n",
       "      <td>southeast</td>\n",
       "      <td>9788.86590</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>438</th>\n",
       "      <td>52</td>\n",
       "      <td>female</td>\n",
       "      <td>46.750</td>\n",
       "      <td>5</td>\n",
       "      <td>no</td>\n",
       "      <td>southeast</td>\n",
       "      <td>12592.53450</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>568</th>\n",
       "      <td>49</td>\n",
       "      <td>female</td>\n",
       "      <td>31.900</td>\n",
       "      <td>5</td>\n",
       "      <td>no</td>\n",
       "      <td>southwest</td>\n",
       "      <td>11552.90400</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>640</th>\n",
       "      <td>33</td>\n",
       "      <td>male</td>\n",
       "      <td>42.400</td>\n",
       "      <td>5</td>\n",
       "      <td>no</td>\n",
       "      <td>southwest</td>\n",
       "      <td>6666.24300</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>877</th>\n",
       "      <td>33</td>\n",
       "      <td>male</td>\n",
       "      <td>33.440</td>\n",
       "      <td>5</td>\n",
       "      <td>no</td>\n",
       "      <td>southeast</td>\n",
       "      <td>6653.78860</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>932</th>\n",
       "      <td>46</td>\n",
       "      <td>male</td>\n",
       "      <td>25.800</td>\n",
       "      <td>5</td>\n",
       "      <td>no</td>\n",
       "      <td>southwest</td>\n",
       "      <td>10096.97000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>937</th>\n",
       "      <td>39</td>\n",
       "      <td>female</td>\n",
       "      <td>24.225</td>\n",
       "      <td>5</td>\n",
       "      <td>no</td>\n",
       "      <td>northwest</td>\n",
       "      <td>8965.79575</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>969</th>\n",
       "      <td>39</td>\n",
       "      <td>female</td>\n",
       "      <td>34.320</td>\n",
       "      <td>5</td>\n",
       "      <td>no</td>\n",
       "      <td>southeast</td>\n",
       "      <td>8596.82780</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>984</th>\n",
       "      <td>20</td>\n",
       "      <td>male</td>\n",
       "      <td>30.115</td>\n",
       "      <td>5</td>\n",
       "      <td>no</td>\n",
       "      <td>northeast</td>\n",
       "      <td>4915.05985</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1085</th>\n",
       "      <td>39</td>\n",
       "      <td>female</td>\n",
       "      <td>18.300</td>\n",
       "      <td>5</td>\n",
       "      <td>yes</td>\n",
       "      <td>southwest</td>\n",
       "      <td>19023.26000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1116</th>\n",
       "      <td>41</td>\n",
       "      <td>male</td>\n",
       "      <td>29.640</td>\n",
       "      <td>5</td>\n",
       "      <td>no</td>\n",
       "      <td>northeast</td>\n",
       "      <td>9222.40260</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1130</th>\n",
       "      <td>39</td>\n",
       "      <td>female</td>\n",
       "      <td>23.870</td>\n",
       "      <td>5</td>\n",
       "      <td>no</td>\n",
       "      <td>southeast</td>\n",
       "      <td>8582.30230</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1245</th>\n",
       "      <td>28</td>\n",
       "      <td>male</td>\n",
       "      <td>24.300</td>\n",
       "      <td>5</td>\n",
       "      <td>no</td>\n",
       "      <td>southwest</td>\n",
       "      <td>5615.36900</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1272</th>\n",
       "      <td>43</td>\n",
       "      <td>male</td>\n",
       "      <td>25.520</td>\n",
       "      <td>5</td>\n",
       "      <td>no</td>\n",
       "      <td>southeast</td>\n",
       "      <td>14478.33015</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      age     sex     bmi  children smoker     region      charges\n",
       "32     19  female  28.600         5     no  southwest   4687.79700\n",
       "71     31    male  28.500         5     no  northeast   6799.45800\n",
       "166    20  female  37.000         5     no  southwest   4830.63000\n",
       "413    25    male  23.900         5     no  southwest   5080.09600\n",
       "425    45    male  24.310         5     no  southeast   9788.86590\n",
       "438    52  female  46.750         5     no  southeast  12592.53450\n",
       "568    49  female  31.900         5     no  southwest  11552.90400\n",
       "640    33    male  42.400         5     no  southwest   6666.24300\n",
       "877    33    male  33.440         5     no  southeast   6653.78860\n",
       "932    46    male  25.800         5     no  southwest  10096.97000\n",
       "937    39  female  24.225         5     no  northwest   8965.79575\n",
       "969    39  female  34.320         5     no  southeast   8596.82780\n",
       "984    20    male  30.115         5     no  northeast   4915.05985\n",
       "1085   39  female  18.300         5    yes  southwest  19023.26000\n",
       "1116   41    male  29.640         5     no  northeast   9222.40260\n",
       "1130   39  female  23.870         5     no  southeast   8582.30230\n",
       "1245   28    male  24.300         5     no  southwest   5615.36900\n",
       "1272   43    male  25.520         5     no  southeast  14478.33015"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#locating row that show people who have max children of 5\n",
    "df.loc[df['children'] == 5.000000]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "age         0.298308\n",
       "bmi         0.198401\n",
       "children    0.067389\n",
       "charges     1.000000\n",
       "Name: charges, dtype: float64"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#check the correlation between charges and each of columns\n",
    "df.corr()[\"charges\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='sex', ylabel='count'>"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#analysis sex vs smoker columns combinely \n",
    "sns.countplot(df[\"sex\"],hue=df[\"smoker\"],palette=\"Set1\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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bBgwzEbnXwQcSVBw7gvEIg0mm2oUQWH5Jc+JzYo6jEhUVGQWEREREREREYmI58RWAq8Ar9jo/DbAyQRTauwUaTApzZ/AQILpMMZGHUakKlCRWURERERERESkCvCc2EzgHuAeG52/BXAIcCiwH1AvymzCj8AHwPvAUM+JFUScR6RKUZFZRERERERERKSK8ZzYVOAx4DEbnV8P6E1QcD4U2CLKbGliEfApQVH5Q8+JzYg4j0iVpiKziIiIiIiIiEgV5jmxFQTFzveB8210/g4ExeZDgO6ovpMoPxF8jT8AhnlObG3EeUSqDf0lJCIiIiIiIiJSjXhO7GfgZ+BOG53fFNgr7rEH0Cy6dNXGWmAMMBz4FvjEc2J/RhtJpPpSkVlEREREREREpJrynNhC4MPwgY3ON6AzQcF5z3C7IxCLKGJV8TtBQbnw8UM4QlxEEkBFZhERERERERGRGsJzYg6MCx/PAdjo/AZAV4KC887ANuGjSTQpkyofmA78AowAvgOGe05sXqSpRGo4FZlFRERERERERGowz4ktA74KH+vY6PzNCIrN2awvPGcDWwP1UxxzU6wCJgOTgIlFtlM8J7Y6wmwiaUlFZhERERERERGRNOQ5sTnAHGBo/PFwyo22BMXmlkDz8NEibj/+0RTIqGScVcBiYBEwH/grfMyN204mKCb/4Tmxgkr2JyIJpCKziIiIiIiIiIisE0658Wf4KJONzs8gWGywEWCb8FhJUFRerNHHItWbiswiIiIiIiIiIlJh4ajieeFDRNJQZT/KICIiIiIiIiIiIiJpTEVmEREREREREREREakwFZlFREREREREREREpMJUZBYRERERERERERGRClORWUREREREREREREQqLDPqACIiIlKzvTrmtagj1DjH79gv6ggiIiIiSWdmtYB9gEOAvYEtgObAXOAb4GF3/zKygCKyjorMIiIiIiIiIiJSFe0LfBLuzwK+B5YB2wHHAsea2S3ufn1E+UQkpOkyRERERERERESkKioA3gD2cfc27n6Yu5/g7jsCJwL5wHVm1jvSlCKiIrOIiIiIiIiIiFQ97v65ux/n7oOLOfcK8Fz49JSUBhORjajILCIiIiIiIiIi1dGocNs+0hQiojmZRURERASO7/9S1BFqlFfzToo6goiISDrYJtzOjDSFiGgks4iIiIiIiIiIVC9m1ho4I3z6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\n",
      "text/plain": [
       "<Figure size 1800x6480 with 8 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#plotting data for visualization\n",
    "\n",
    "catergoy_column=[\"sex\",\"smoker\",\"region\",\"children\"]\n",
    "\n",
    "colors=[\"#00FFFF\",\"#FFA597\",\"#00CFFC\",\"#ED00D9\",\"#ADD8E6\",\"#EFF999\"]\n",
    "textprops = {\"fontsize\":22}\n",
    "\n",
    "plt.figure(figsize=(25,90))\n",
    "i=1\n",
    "for column in catergoy_column:\n",
    "    plt.subplot(11,2,i)\n",
    "    sns.countplot(data=df,x=column,palette=\"Spectral\")\n",
    "    plt.xticks(fontsize=25)\n",
    "    plt.yticks(fontsize=25)\n",
    "    plt.xlabel(column,fontsize=25)\n",
    "    plt.ylabel(\"count\",fontsize=25)\n",
    "    i=i+1\n",
    "    plt.subplot(11,2,i)\n",
    "    df[column].value_counts().plot(kind=\"pie\",autopct=\"%.2f%%\",colors=colors,textprops=textprops,radius = 1.1)\n",
    "    plt.xticks(fontsize=25)\n",
    "    plt.yticks(fontsize=25)\n",
    "    plt.xlabel(column,fontsize=25)\n",
    "    plt.ylabel(\"count\",fontsize=25)\n",
    "    i=i+1\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='charges', ylabel='Density'>"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAXgAAAERCAYAAABxZrw0AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8vihELAAAACXBIWXMAAAsTAAALEwEAmpwYAAAqtUlEQVR4nO3deXxcZ33v8c9Po321LI1kW453eYuz2DGxsxljkpAFwlLoJWErhYYCpVDKZSm9tNx7e9veS7ktt0CTQsoSCDQkhKRlSXAWZ7UjJ3G8xpa3eNVi2Vps7fO7f8yIyIpsS7LOnJnR9/16zUsz55w55+ux/ZtHz3nOc8zdERGRzJMVdgAREQmGCryISIZSgRcRyVAq8CIiGUoFXkQkQ6nAi4hkqJQr8GZ2l5k1mtmWcdpfv5m9lHg8OB77FBFJB5Zq4+DNbBXQAfzA3ZeMw/463L34/JOJiKSXlGvBu/s6oGXwMjOba2a/NrONZvakmS0MKZ6ISNpIuQJ/BncCn3L3y4DPAd8axXvzzazOzJ4zs3cEkk5EJAVlhx3gXMysGLgSuNfMBhbnJda9C/jvw7ztkLu/JfF8hrsfNrM5wKNmttnddwedW0QkbClf4In/lnHC3S8dusLd7wfuP9ub3f1w4uceM3scWAqowItIxkv5Lhp3bwP2mtl7ACzukpG818zKzWygtV8JXAVsCyysiEgKSbkCb2b3AM8CC8zsoJl9BHgf8BEz2wRsBd4+wt0tAuoS73sM+Dt3V4EXkQkh5YZJiojI+Ei5FryIiIyPlDrJWllZ6bNmzQo7hohI2ti4cWOzu0eHW5dSBX7WrFnU1dWFHUNEJG2Y2f4zrVMXjYhIhlKBFxHJUCrwIiIZSgVeRCRDqcCLiGQoFXgRkQylAi8ikqFU4EVEMpQKvIhIhkqpK1nT3Y/Xv3rW9betmJGkJCIiasGLiGQsFXgRkQylAi8ikqFU4EVEMpQKvIhIhlKBFxHJUCrwIiIZKrACb2YLzOylQY82M/tMUMcTEZHTBXahk7u/AlwKYGYR4BDw86COJyIip0tWF82bgd3ufsZ7B4qIyPhKVoF/L3DPcCvM7HYzqzOzuqampiTFERHJfIEXeDPLBW4B7h1uvbvf6e7L3X15NBoNOo6IyISRjBb8jcAL7t6QhGOJiEhCMgr8rZyhe0ZERIITaIE3s0LgOuD+II8jIiKvF+h88O5+CqgI8hgiIjI8XckqIpKhVOBFRDKUCryISIZSgRcRyVAq8CIiGUoFXkQkQ6nAi4hkKBV4EZEMpQIvIpKhVOBFRDKUCryISIZSgRcRyVAq8CIiGUoFXkQkQwU6XbCc7sfrXz3r+ttWzEhSEhGZCNSCFxHJUCrwIiIZSgVeRCRDqcCLiGSooG+6PcnMfmZmO8xsu5ldEeTxRETkNUGPovkn4Nfu/m4zywUKAz6eiIgkBFbgzawUWAX8AYC79wA9QR1PREROF2QXzRygCfg3M3vRzL5jZkVDNzKz282szszqmpqaAowjIjKxBFngs4FlwLfdfSlwEvji0I3c/U53X+7uy6PRaIBxREQmliAL/EHgoLuvT7z+GfGCLyIiSRBYgXf3o8ABM1uQWPRmYFtQxxMRkdMFPYrmU8CPEiNo9gAfDvh4IiKSEGiBd/eXgOVBHkNERIanK1lFRDKUCryISIZSgRcRyVAq8CIiGUoFXkQkQ6nAi4hkKBV4EZEMpQIvIpKhgr6SVUagtz/Gul1N/GzjAY60djGvqpi3XTyNdy2rITui72ARGRtVj5A1t3fzrcfrWbu9ETNj5ZwKjrR28fn7XuaGf3qSV462hx1RRNKUWvAh6uzp53vP7qOrt58/uHIWf33LhQC4Ow9va+AvH9jCO7/1NN+8bRlvWlgVcloRSTdqwYfE3bl34wFOnOrhAytnMr+65HfrzIy3XDiFh/7kauZEi/jYDzfy5C7dDEVERkcFPiTbj7Sx42g7Ny6ZysyK193oCoApZfnc/ZEVzK0q5o9+UMfmg61JTiki6UwFPgSxRBdMtDiPlXMqzrrtpMJcfviRy6koyuNjP6yjuaM7SSlFJN2pwIfgpQMnaGzv5trF1USy7JzbVxbncccHLqPlVA9/es+LxGKehJQiku5U4JPM3Xm6vpkppfksmVY64vctqSnjq7dcyDO7j/Gdp/YEmFBEMoVG0STZ4RNdHGnt4pZLpmF2euv9x+tfPet7b738Ah7d0cj/+c0rrJofZeGUkX9BiMjEoxZ8kj2/v4XsLOOS6ZNG/V4z42/fdTEl+Tl84b7N9KurRkTOQgU+iXr6Ymw6cIKLasooyI2MaR+Ti3L5q7ctZtOBE/zg2X3jG1BEMkqgBd7M9pnZZjN7yczqgjxWOtjV2E53X4ylM8rPaz+3XDKNVfOjfP3hnTS1a1SNiAwvGX3wb3L35iQcJ+VtP9JGfk4WsyuHH/d+LoP76C+bUc5Tu5r4+N0bedey6b9bftuKGeedU0Qyg06yJkl/zNlxtJ2FU0pHNDTyXKIleVw5t5Kn65tZOaeCaZMKgHOfqNUXgMjEEXQfvAMPm9lGM7t9uA3M7HYzqzOzuqamzL0c/9WWU5zq6WfR1PEb+bJmYRX5OREe3nZ03PYpIpkj6AJ/lbsvA24EPmlmq4Zu4O53uvtyd18ejUYDjhOe7UfaiGQZ86uKx22f+TkRVi+IsrOhgz1NHeO2XxHJDIEWeHc/nPjZCPwcuDzI46WyXY3tzK4oIi9nbKNnzmTlnApK87N5eFsD7ho2KSKvCazAm1mRmZUMPAeuB7YEdbxU1tHdR0NbN3OiYzu5ejY5kSxWL6ji1ZZT7Gk+Oe77F5H0FWQLvhp4ysw2ARuA/3T3Xwd4vJQ10H0yJzp+3TODXTaznJL8bB7b0RjI/kUkPQU2isbd9wCXBLX/dLKn+SR52VnUJEa6jLecSBbX1Eb55eYj7D928ozTD4vIxKIrWZNgT1MHsyqKxmV45JlcPmsyBTkRnqrXJQciEqcCH7C2zl6aO3oC6X8fLDc7i8tnT2bb4TZaTvYEeiwRSQ8q8AHb33IKgFlJ6DZZOacCM3hmt1rxIqICH7gDLafIzjKmTsoP/FhlBTlcPH0SG/cfp7uvP/DjiUhqU4EP2Kstp5g2qYDsrOR81CtmT6a7L8bLun+ryISnAh+gvliMwyc6mTG5MGnHnDG5kCml+WzY25K0Y4pIalKBD9DR1i76Ys4FSSzwZsblsydz6EQnB4+fStpxRST1qMAH6NXECdYLyoMZ/34ml14wiZyIsXH/8aQeV0RSiwp8gA60nKI0P5uygpykHjc/J8KF08rYdPAEvf2xpB5bRFLHiAq8md1nZjebmb4QRuHQiS5qygtfd3PtZFg2o5yu3hjbj7Ql/dgikhpGWrC/DdwG7DKzvzOzhQFmygjdvf0c6+imJgnDI4czJ1pEWUEOL7yqbhqRiWpEBd7df+vu7wOWAfuAR8zsGTP7sJklt/8hTRxp7cKBaWXJ7X8fkGXGpRdMor6xg5PdfaFkEJFwjbjLxcwqgD8APgq8CPwT8YL/SCDJ0tzh1k6A391KLwwX1ZQRc9imbhqRCWmkffD3A08ChcDb3P0Wd/+pu38KCGYO3DR3+EQnxXnZlOSHd9vbqWX5TC7KZcshXfQkMhGNtPp8x91/OXiBmeW5e7e7Lw8gV9o7fKKLaZPyQznBOsDMuKimjCd3NXGqu4/CPN1jXWQiGWkXzf8cZtmz4xkkk/T2x2hs7wqt/32wJYlumq3qphGZcM7apDOzKUANUGBmS4GB5mgp8e4aGUZDWxcxh6kh9r8PmDaom+YNsyaHHUdEkuhcv7O/hfiJ1enA1wctbwf+IqBMae9oaxcQL65hG9pNIyITx1kLvLt/H/i+mf2eu9+XpExp70hbFzkRo7woN+woQLyb5omdTRpNIzLBnKuL5v3ufjcwy8w+O3S9u399mLcN3UcEqAMOuftbx5w0jTS0dlFdmk9WiCdYBxvoptms0TQiE8q5TrIO3IaoGCgZ5jESnwa2jyldGnJ3jrZ1MaU0/O6ZAWbGkmll7G7qoK2rN+w4IpIk5+qiuSPx86tj2bmZTQduBv4GeN1vAJmovbuPUz39TEmB/vfBFk4pYd2uJp7c2czNF08NO46IJMFIL3T632ZWamY5ZrbWzJrN7P0jeOs/Ap8HzjiloZndbmZ1ZlbX1NQ0stQprCFxgrU6hVrwABdMLqQgJ8KjOxrDjiIiSTLScfDXu3sb8FbgIDAf+K9ne4OZvRVodPeNZ9vO3e909+XuvjwajY4wTuo62hYv8KnURQMQyTLmVxfz+CuNxGIedhwRSYKRFviBCcVuAu5x95HcD+4q4BYz2wf8BFhjZnePPmJ6OdraRUl+NkUpeNXogimlHDvZw6aDJ8KOIiJJMNIq9JCZ7QA6gU+YWRToOtsb3P1LwJcAzGw18Dl3H0m3TlpraO9Kue6ZAfOrizHgG2vruW5x9bDb3LZiRnJDiUhgRjpd8BeBK4Dl7t4LnATeHmSwdBRzp6m9m6qSvLCjDKswN5sZFYW8clTj4UUmgtH0IywiPh5+8Ht+MJI3uvvjwOOjOFZaaj3VS2+/E03RAg+wsLqE32xroK2zl9Ik30pQRJJrpKNofgh8DbgaeEPioVkkh2hs7wagqiQ1u2gAFkwtBeCVhvaQk4hI0Ebagl8OLHZ3Db84i6b2+GmJVO2iAaguyWNSQQ47jrZr8jGRDDfSUTRbgClBBskEje3dFOZGUnIEzQAzY8GUEnY3dtDXf8bLE0QkA4y0ElUC28xsA9A9sNDdbwkkVZpK5ROsg82vLmH93hb2t5xiblQ35BLJVCMt8H8dZIhM4O40tnezpKYs7CjnNCdaRMSMXQ0dKvAiGWykwySfAPYBOYnnzwMvBJgr7Rw72UNnb39atODzsiPMqChkV6NOtIpkspGOovkj4GfAHYlFNcADAWVKS/WNHQApPURysNqqYo60dtGu2SVFMtZIT7J+kvjUA20A7r4LqAoqVDoaKPDp0IIHqK2Oz/Y8kFtEMs9IC3y3u/cMvEhc7KQhk4PUN3aQG8miLE0uHppalk9RboRdKvAiGWukBf4JM/sL4jffvg64F3gouFjpZ3dTB9GSPCxF7uJ0LllmzKsqZldjBzFd3iCSkUZa4L8INAGbgY8BvwT+MqhQ6ai+sSNt+t8H1FaXcLK773c3CReRzDKiYZLuHjOzB4AH3D3978oxzjq6+zjS2sVFaTBEcrB5VfEhkrsaO5g2qSDkNCIy3s7agre4vzazZmAH8IqZNZnZV5ITLz3sTrMRNANK83OYUprPLs1LI5KRztVF8xnio2fe4O4V7j4ZWAFcZWZ/FnS4dJFuQyQHq60qZv+xU/T0adoCkUxzrgL/QeBWd987sMDd9wDvT6wToL6pg+wso6IoDQt8dQn97uxp1mgakUxzrgKf4+7NQxcm+uHTYzxgEtQ3djCrsohIVnqMoBlsZkUhOZH4tAUiklnOVeB7xrhuQtnd2MG8NJ3TJSeSxezKIo2HF8lA5yrwl5hZ2zCPduCiZARMdT19Mfa3nPrdiJR0VFtVQnNHN8dP6TtbJJOcdZiku0eSFSRd7Tt2kv6YM6+qmFM9/WHHGZPageGS6qYRySgjvdBp1Mws38w2mNkmM9tqZl8N6lhhGhhBk84t+GhJHmUFOZpdUiTDBFbgid8YZI27XwJcCtxgZisDPF4oBgr8nGhRyEnGzsyorSpmd5Pu8iSSSQIr8B438Dt/TuKRcZOe1Dd2UDOpgMLc1L1N30jMqyqmqzfGpoOtYUcRkXESZAseM4uY2UtAI/CIu68fZpvbzazOzOqamtJvFoT6xo607p4ZMK+qGAPW7Uy/vwMRGV6gBd7d+939UmA6cLmZLRlmmzvdfbm7L49Go0HGGXexWPwCoUwo8IW52UwvL2DdLhV4kUwRaIEf4O4ngMeBG5JxvGQ5dKKTrt5YRhR4gHlVJWw6cILWU7rLk0gmCHIUTdTMJiWeFwDXEp+wLGNkwgiaweZXFxNzeHr36y5eFpE0FGQLfirwmJm9TPwm3Y+4+38EeLyk+12BT9OrWIeaXl5ISV62+uFFMkRgQz/c/WVgaVD7TwX1jR1UFOVSXpQbdpRxEckyrpxXwZO7mnH3tLk7lYgMLyl98JmqvqmDuRnSPTNg1fwoh050srvpZNhRROQ8qcCPkbtnzBDJwVbVxkcyqZtGJP2pwI9Rc0cPrZ29GdP/PuCCyYXMriziSQ2XFEl7KvBjlGkjaAZbVVvJc3ta6O5Lz8nTRCROBX6M6psyt8BfUxuls7efun3Hw44iIudBBX6Mdjd2UJQbYWpZfthRxt0VcyvIiZiuahVJcyrwY1TfGB9Bk4lDCYvyslk2o5x1O3XBk0g6U4Efo/o0vk3fSKyaH2X7kTYa27vCjiIiY6QCPwatnb0cbeuitrok7CiBWb0gPlzysR2NIScRkbFSgR+D+sSdj2oz8ATrgMVTS6mZVMDDWxvCjiIiY6QCPwY7E/cunZ/BLXgz47rF1TxV38ypnr6w44jIGKjAj8Guhg4KciJMLy8IO0qgrl9cTXdfTCdbRdKUCvwY7GpsZ15VMVlZmTeCZrA3zJ5MaX42D287GnYUERkDFfgx2NnQTm115va/D8iJZHHt4moe2dZAT59uxi2SblTgR6m1s5eGtu6M7n8f7K0XT6W9q4+n6nXRk0i6UYEfpV0N8RE08ydACx7g6nlRSvKz+Y+Xj4QdRURGSQV+lAZG0NRWTYwWfG52Fm+5cAqPbG3Q5GMiaUYFfpR2NrRTkBOhZlJmj6AZ7OaLp9Le3cfjr6ibRiSdqMCPUn1jB7XVmT+CZrCr51VSWZzL/S8cDDuKiIxCYAXezC4ws8fMbLuZbTWzTwd1rGTa2dA+YbpnBuREsnj7pTU8uqOR4yd7wo4jIiMUZAu+D/hzd18ErAQ+aWaLAzxe4FpP9dLY3j1hTrAO9q5lNfT2Ow+9fDjsKCIyQoEVeHc/4u4vJJ63A9uBmqCOlww7GwdG0EysFjzAhdPKWDilhHvr1E0jki6S0gdvZrOApcD6YdbdbmZ1ZlbX1JTaJ/F2JoZIToSLnIZz24oZbD7UyqYDJ8KOIiIjEHiBN7Ni4D7gM+7eNnS9u9/p7svdfXk0Gg06znnZ1RC/i9NEGkEz2DuX1lCYG+GHz+0PO4qIjECgBd7McogX9x+5+/1BHisZdjbE56DJxLs4jURJfg7vXFrDQ5sO62SrSBoIchSNAd8Ftrv714M6TjLtauzI6Jt8jMQHrphJd1+MH61XK14k1QXZgr8K+ACwxsxeSjxuCvB4gWru6KapvZuFUyZ2gV84pZTVC6L829P76OzRla0iqSzIUTRPubu5+8Xufmni8cugjhe07Ufipw8WTy0NOUn4PrF6HsdO9vDvdQfCjiIiZ6ErWUdooMAvUoHn8tmTWT6znH95YjddvWrFi6QqFfgR2n6knSml+ZQX5YYdJSV89vr5HGnt4vvP7As7ioicgQr8CG073MbiaWq9D7hybiWrF0T55mP1nDilETUiqUgFfgS6evvZ3dTBoqkT+wTrUF+4YSEd3X187eFXwo4iIsNQgR+B+sYO+mKu/vchFk0t5Q+unM3dz71K3b6WsOOIyBAq8COwTSNozujPr59PzaQCvnDfyxo2KZJissMOkA62HW6jMDfCzIqisKME7sfrXz3r+ttWzDjtdVFeNn//exfzgbvW89cPbuXv331xkPFEZBTUgh+Blw+eYMm0MiIT6CYfo3F1bSWfXD2Pn9Yd0Nh4kRSiAn8Off0xth5u46LpZWFHSWmfubaWq+dV8hf3b+bp+uaw44gIKvDntKuxg+6+GBerwJ9VdiSLb71/GXOiRXzshxvZuP942JFEJjz1wZ/D5oOtAFxUowJ/LqX5Obxz6XS+8+Qebv3X5/jgypnMib5+7vyh/fgiEgy14M/h5UMnKMnLZtYEOME6HsoKcvjoNXMoK8jhrqf38tyeY2FHEpmwVODPYfPBVpbUlJGlE6wjVlaQw8ffOJfaqhIe3HSYB148RF8sFnYskQlHBf4suvv62X6kXSdYxyA/J8IHrpjJG+dH2bCvhTue2ENDW1fYsUQmFBX4s9h6uI2e/hjLZkwKO0payjLjLRdO4X0rZnD8VA/ffKyep3Y1EYt52NFEJgSdZD2LjfviI0GWzSwPOUnqONeFUMO5cFoZMyYX8sBLh/nllqO8545n+dt3XcT8CX53LJGgqQV/Fhv3H2fG5EKqSvLDjpL2SvJzeP+KGbznsunsaerg5m88ydd+84rmkxcJkFrwZ+Du1O0/zqrayrCjZAwzY+mMcuZXl/CrLUf458fquWfDq7z90hrmVb02nFLDKEXGh1rwZ3CgpZPmjm51zwSgKC+bd192AX941WwA7np6L/fWHeBkd1/IyUQyS2AF3szuMrNGM9sS1DGCVLc/Pv3t8lkq8EGZV1XMn765ltULomw6eIL/+9udvLD/OO46CSsyHoJswX8PuCHA/Qfq+X0tlORnU1ulE4FByolkcf3iKXxqTS2VxXn87IWDvO8769nbfDLsaCJpL7AC7+7rgLS9C8TT9cdYMbtCM0gmSXVpPrevmsPbL53G5kOtvOUf1/H/1u6ip08XSImMVeh98GZ2u5nVmVldU1NT2HEAONByildbTnHVvIqwo0woWWasmF3B2s++kesWVfMPj+zk5m88qbtFiYxR6AXe3e909+XuvjwajYYdB4Bndsenu71qnkbQhKGqNJ9vvm8Z3/3Qck719PPuf3mWL92/mdbO3rCjiaQVDZMcxtP1x4iW5FFb9fqZECV4gy+m+ug1s/nttgZ+suFVHtp0mJsvmsrf/d5FmKnrTORcQm/Bpxp355ndx7hyboWKSArIy45w88XT+MSb5lFWkMNP6w7wwbs2sE8nYUXOKbAWvJndA6wGKs3sIPBX7v7doI43XrYebqO5o1vdMymmZlIBH189l/V7W3h8RyPX/+M6/uRN8/jYG+eQlx0JO96YjPb+tyKjFViBd/dbg9p3kB7Z1oAZrFlYFXYUGSLLjCvmVPCXNy/if/zHNr7+yE4eeOkQf/OOi7hibmqdEB/LnD0i40198EP8dnsDl80op7I4L+wocgbVpfn8823LePdljXzlF1u59V+f413LavjyTYuoSOG/t77+GEfbumju6Kajq4+O7j56+mNEzMjKMnKzs5hUkENZQS7lhTnEYq77EMh5UYEf5PCJTrYebuOLNy4MO4qMwOoFVTz8ZxX886P13LFuN2u3N/L5Gxbw3jfMCPz6hZG00Ptjzv5jJ9l+pI29x07S0NpN/6CrdCMWL+oxd/pjTt+QaZTvWLeHRVNLWFJTxuWzJnP57Mkp/QUmqUcFfpC12xsAuHZRdchJ5GyGFtdpkwr4xOp5PLjpMF/++Rbufu5V/upti1k5J5xum8MnOlm/9xhbDrXR2dtPdpYxs6KQq2srqZlUQFVJHiX5OeTnZJ12Ir+vP0ZbVx8nOns41tFDSX42Ww+3cc+GV/m3p/cBML+6mJVzKrhybgVXzqukND8nlD+jpAdLpXk/li9f7nV1daEd//fveJbmjm4e/fPVY3q/+l3D5e5sOdzGrzYf4URnL0tqyrhxyRTKC3Nft+35nsAc+nfdF4ux9VAbz+45xqstp8iJGBdOK2Px1FJqq4vP60RwXyzGoeOd7G0+yd7mk+w7dpLefieSZSybMYk3zo+yan6UJdN0a8mJyMw2uvvy4dapBZ9woOUUG/a28Lnr54cdRcbIzLiopoyFU0p4clcTT+xsYseRNlbMnsyq+VFKAmjttnb2smHvMZ7fd5yO7j4qinK56aKpXDajnILc8Rndk52VxcyKImZWFLF6QbzrZ+BLZN2uJr728E6+9vBOJhflck1tJatqo1wzv1L3MRAV+AEPvHgIgHcsrQk5iZyvnEgWaxZWs2xGOWu3N/LsnmNs2NfCytkVXDG3gknDtOhHIxZzdjW2s2FvC9uPtOEOC6aUsHJOBfOqiskK+PqJSJYxu7KI21bM4PM3LKSpvZun6ptYt7OZdTub+MVLhwFYPLWUNy6Isqo2ymUzy8nN1mUvE426aIj/ar/mH56gujSPn9x+xZj3oy6a1NTc0c2jOxrZdOAEAIumlvKlmxZyxZwKsiMjL3r7j53koU2H+WndAQ60dFKQE+ENs8q5fHYFk4vO70tjvMTcOdLaxa6Gdk509vLC/uP0xZySvGyuXVzNjUumsGp+lPycyIj+vWosfupTF805PLenhb3NJ/n46rlhR5EAVBbn8fvLL+C6RdWs39tC3f4WPvDdDZTmZ3NNbZRV8yu5cFoZF0wupDQ/GzOjq7efwyc62dnQzvP7jvPYjkb2JK6evWJOBVfOqWTxtFJyRvEFkQxZZtRMKqBmUgG3rZhBe1cvz+w+xm+3NfDwtgZ+/uIhinIjrFlUTXlhDvOrSwL/jUPCowIPfPepvVQU5XLLJdPCjiIBKi/K5YYlU3jzoiqiJXk88UoTj+9s5D83H/ndNgPnKAePWMzNzmLlnAo+eMVM3ryomgsmF6bFb2uDMy6dUc7F0yexp6mDLYdbWbu9gVM9/ZTmZ3PZzMm8YVb5eXddSeqZ8AV+T1MHa3c08KdrasnPSc9L3mV0ciJZ3HTRVG66aCruzu6mk+xsaOfQ8U5aO3sxg/ycCNWl+cyrKmZBdcm4nTANUyTLqK0uoba6hLddEmPHkXbq9rfw+CuNPLGzkYunT+LqeZVMm1QQdlQZJxO+wP/rk3vJiWTx/pUzw44iITAz5lUVn3bT74kgOyuLJTVlLKkp4/jJHp7Z3czz+47z0oETzK8u5tpF1UwvLww7ppynCV3g6xvb+fe6A7x/xQyiJbpCUCam8qJcbr54GmsWVrN+7zGeqm/mW4/vZtGUEi65oIwLp5WFHVHGKLXOECXZ3/5yB4U5ET59rca+ixTkRli9oIrPXb+AaxdVs/fYSW7+xlN8/O6NvHK0Pex4MgYTtgX/yLYG1u6Iz12SKkPcRFJBfk6ENQuruGJOBa1dvdz11F5+vfUob714Gp+5tpa50YnVnZXOJmSBb2zv4gv3vcziqaV85OrZYceREKTDKJiwFeRG+Mg1s/nwlbP41yf38L1n9vGfLx/mHUtr+PSba5lZURR2RDmHCddF093Xz2d+8hInu/v4xq2Xpu3NIkSSpbwol8/fsJB1n38TH7l6Nv/58hHW/MMTfPG+l9nT1BF2PDmLCdWC7485n/3pJp7ZfYyv//4lzKsqCTuSSEob+pvO7Mpi/uy6+Tyxs4l7Nx7kJ88f4E0Lonz4qtlcPa9Sk52lmAlT4Nu6evnUj1/kiZ1N/MVNC3nXsulhRxJJS6X5Obzt4mmsnh+ls7efu5/bzwfv2kDNpALefuk03rG0hvnVajylgglR4B/b0chXHtzCkRNd/M07l/C+FRrzLnK+SvJz+Ngb5/Lx1XP51eaj/PzFQ/zLE7v51uO74/Pfz6vkmtooy2ZO0syWIQm0wJvZDcA/ARHgO+7+d0Eeb7D2rl4e3dHID57dz8b9x5lXVcw9t6/kDbMmJyuCyISQlx3hHUtreMfSGprau/nVliOs29nEAy8e4keJLp5oSR4XTitlbrSY6eUFTC8vJFqSR1lBDmUFOZTmZ49q4jcZmcAKvJlFgG8C1wEHgefN7EF33zaex4nFnG1H2jjQcoqDxzs5cPwUmw+1suVQK739zozJhXz1lgu59fIZmi5VZJwNNxopOys+XfMb51exaGoJLx9sZcvhVrYdbmP9nhY6e/uH3VdhboScSBbZWUZ2xMjOyiI7YkSyDCN+1XH8Z3x7w157PmidWXwdidctJ3sS27+2LcQnZsuOGDMnF5KbnRV/RLJee56dRd7g15EscrMjp73Oyx66/szrwvgCC7IFfzlQ7+57AMzsJ8DbgXEt8Gbwe99+hu6+GAAledksnFrCH149m+sWxecE14kfkeSLZBnLZ01m+aDfmt2dYyd7ONByipaTPfx6y1E6e/vp7O2nuzdGf8zpdycWc2LuxDw+OMLjb2bw5OYDM50PrIslXsSXe2KdU5gbGbJtfHnMobu3n6NtXfT0xejpj8V/Jh7didfjJcviX37Y4C+b+M/K4jye+sKacTvWgMDmgzezdwM3uPtHE68/AKxw9z8Zst3twO2JlwuAV4bsqhJoDiTk+VGu0UvVbMo1Oso1OkHnmunu0eFWBNmCH67Z/LpvE3e/E7jzjDsxqzvTZPZhUq7RS9VsyjU6yjU6YeYKslPoIHDBoNfTgcMBHk9ERAYJssA/D9Sa2WwzywXeCzwY4PFERGSQwLpo3L3PzP4E+A3xYZJ3ufvWMezqjN03IVOu0UvVbMo1Oso1OqHlSqmbbouIyPjRwHARkQylAi8ikqncPakP4D3AViAGLB+y7ktAPfGx8G8ZtPwyYHNi3Td4rWspD/hpYvl6YNag93wI2JV4fGic/ww3JDLWA18M6HO6C2gEtgxaNhl4JPFnegQoD+KzO0umC4DHgO2Jv8NPp0iufGADsCmR66upkGvQPiPAi8B/pFiufYl9vgTUpUo2YBLwM2BH4t/aFWHnIn6NzkuDHm3AZ8LOdc7c41GMRnVAWJT4sB5nUIEHFhP/D5oHzAZ2A5HEug2Jv2QDfgXcmFj+CeBfEs/fC/x00D/SPYmf5Ynn5eOUP5LINgfITWReHMDntApYxukF/n+T+EIBvgj8/Xh/dufINBVYlnheAuxMHDvsXAYUJ57nJP5zrAw716B8nwV+zGsFPlVy7QMqhywLPRvwfeCjiee5xAt+6LmG1ICjwMxUyjVs1vPdwZgP/PoC/yXgS4Ne/ybxIUwFdgxafitwx+BtEs+ziV8tZoO3Say7A7h1nHJfAfzmTLnH+TOaxekF/hVgauL5VOCV8f7sRpnvF8TnGkqZXEAh8AKwIhVyEb/+Yy2whtcKfOi5Etvv4/UFPtRsQCmwd+h2YecakuV64OlUyzXcI5X64GuAA4NeH0wsq0k8H7r8tPe4ex/QClScZV9B5kyGanc/ApD4WXWOTGP57EbEzGYBS4m3lkPPZWYRM3uJeLfWI+6eErmAfwQ+T7xLckAq5IL4leUPm9nGxJQhqZBtDtAE/JuZvWhm3zGzohTINdh7gXsSz1Mp1+sEUuDN7LdmtmWYx9vP9rZhlvlZlo/1PecryH2P1Xh+duc+mFkxcB/wGXdvS4Vc7t7v7pcSbzFfbmZLws5lZm8FGt1949m2S3auQa5y92XAjcAnzWxVCmTLJt41+W13XwqcJN71EXau+BvjF23eAtx7rk2TmetMAinw7n6tuy8Z5vGLs7ztTFMbHEw8H7r8tPeYWTZQBrScZV/jIcwpGBrMbCpA4mfjOTKN5bM7KzPLIV7cf+Tu96dKrgHufoJ4998NKZDrKuAWM9sH/ARYY2Z3p0AuANz9cOJnI/Bz4jPAhp3tIHAw8RsYxE+2LkuBXANuBF5w94bE61TJNaxU6qJ5EHivmeWZ2WygFtiQ+LWn3cxWWnwi5w8S7/sdeM+HEs/fDTzq8Q6s3wDXm1m5mZUT7zP7zTjlDHMKhsF/3g9x+ucwXp/dGSX28V1gu7t/PYVyRc1sUuJ5AXAt8REYoeZy9y+5+3R3n0X838mj7v7+sHMlPqciMysZeE78/8iWsLO5+1HggJktSCx6M/EpxkP/zBJu5bXumaH7CjPX8M6nA38sD+CdxL+puoEGTj9h+WXiZ5tfIXFmObF8OfF/fLuBf+a1YUX5xH9Vqid+ZnrOoPf8YWJ5PfDhcf4z3ER8BMlu4MsBfU73AEeA3sTn9RHi/XFriQ/JWgtMDuKzO0umq4n/yvgyrw0XuykFcl1MfBjiy4l9fiWxPNRcQzKu5rWTrKHnIt7XvYnXhpZ+OYWyXQrUJf4+HyA+Ei4VchUCx4CyQctCz3W2h6YqEBHJUKnURSMiIuNIBV5EJEOpwIuIZCgVeBGRDKUCLyKSoVTgZUIxs++Z2bvDziGSDCrwIiNkcfo/I2lD/1glo5nZB83sZTPbZGY/TCxeZWbPmNmegda8mRWb2Voze8HMNg/Mm2Rms8xsu5l9i/gslReY2X8zsx1m9oiZ3WNmn0tsO9fMfp2YvOtJM1uYWP6exFxMm8xsXQgfg0xQutBJMpaZXQjcT3xSrWYzmwx8HSgC/guwEHjQ3ecl5v4odPc2M6sEniN+eflM4vcTuNLdnzOz5cB3iE/9mk286N/h7l8zs7XAH7v7LjNbAfytu68xs83ADe5+yMwmeXy+HJHAZYcdQCRAa4CfuXszgLu3xKf/4AF3jwHbzKw6sa0B/8viMyrGiE/dOrBuv7s/l3h+NfALd+8EMLOHEj+LgSuBexPHgPjNHgCeBr5nZv9O/AtHJClU4CWTGcNPt9o9ZBuA9wFR4DJ3703MAJmfWHdymO2HygJOeHzK4tO4+x8nWvQ3Ay+Z2aXufmzEfwqRMVIfvGSytcDvm1kFQKKL5kzKiM/d3mtmbyLeNTOcp4C3mVl+otV+M4DH58Xfa2bvSRzLzOySxPO57r7e3b9C/C49F5xh3yLjSi14yVjuvtXM/gZ4wsz6ic84eSY/Ah4yszris2TuOMM+nzezB4nPwrif+KyHrYnV7wO+bWZ/SfzesD9JbPd/zKyWeOt/bWKZSOB0klVklMys2N07zKwQWAfc7u4vhJ1LZCi14EVG704zW0y8j/77Ku6SqtSCFxHJUDrJKiKSoVTgRUQylAq8iEiGUoEXEclQKvAiIhnq/wOnPV3ICaZkzgAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Check for distribution of charges\n",
    "sns.distplot(df['charges'])\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "q = df['charges'].quantile(0.99)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1296x576 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#analysis age and bmi columns\n",
    "plt.figure(figsize=(18,8))\n",
    "plt.subplot(121)\n",
    "sns.histplot(df[\"age\"],color=\"#ED00D9\",fill=True)\n",
    "plt.title(\"Age\")\n",
    "plt.xticks(fontsize=25)\n",
    "plt.yticks(fontsize=25)\n",
    "plt.xlabel(\"Age\",fontsize=25)\n",
    "plt.ylabel(\"count\",fontsize=25)\n",
    "    \n",
    "\n",
    "plt.subplot(122)\n",
    "sns.histplot(df[\"bmi\"],color=\"#00FFFF\")\n",
    "plt.title(\"BMI\")\n",
    "plt.xticks(fontsize=25)\n",
    "plt.yticks(fontsize=25)\n",
    "plt.xlabel(\"BMI\",fontsize=25)\n",
    "plt.ylabel(\"count\",fontsize=25)\n",
    "   \n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='age'>"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#checking for outliers in age col\n",
    "\n",
    "sns.boxplot(df[\"age\"],palette=\"Spectral\")\n",
    "            \n",
    "#NO outliers in AGE\n",
    "            "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='bmi'>"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#outliers in bmi col...\n",
    "sns.boxplot(df[\"bmi\"],palette=\"Spectral\")\n",
    "           \n",
    "           \n",
    "#There are outliers in BMI\n",
    "\n",
    "#Outliers increase the variability in your data, which decreases statistical power. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='bmi', ylabel='Density'>"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Check for distribution of BMI\n",
    "sns.distplot(df['bmi'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "columns name : bmi\n",
      "numbers of outliers: 4\n",
      "\n",
      "\n",
      "      age   sex    bmi  children smoker     region     charges\n",
      "116    58  male  49.06         0     no  southeast  11381.3254\n",
      "847    23  male  50.38         1     no  southeast   2438.0552\n",
      "1047   22  male  52.58         1    yes  southeast  44501.3982\n",
      "1317   18  male  53.13         0     no  southeast   1163.4627\n",
      "<<<<<<<<<------------------------------------->>>>>>>>>\n"
     ]
    }
   ],
   "source": [
    "#REMOVING OUTLIERS TO TRAIN DATA\n",
    "\n",
    "\n",
    "\n",
    "def outlier(data):\n",
    "\n",
    "    mean=data.mean()\n",
    "    std=data.std()\n",
    "    mini=data.min()\n",
    "    maxi=data.max()\n",
    "\n",
    "    #let find the boundaries for outlier\n",
    "    highest=data.mean() + 3*data.std()\n",
    "    lowest=data.mean() - 3*data.std()\n",
    "\n",
    "        #finally, let find the outlier\n",
    "    outliers=df[(data>highest) | (data<lowest)]\n",
    "        \n",
    "\n",
    "    return outliers\n",
    "#outliers detection and remove  \n",
    "new=pd.DataFrame(df[\"bmi\"],columns=[\"bmi\"])\n",
    "for col in new.columns:\n",
    "    test=outlier(df[col])\n",
    "    print(\"columns name :\",col)\n",
    "    print(\"numbers of outliers:\",len(test))\n",
    "    print(\"\\n\")\n",
    "    print(test)\n",
    "    print(\"<<<<<<<<<------------------------------------->>>>>>>>>\")\n",
    "    \n",
    "#drop the outliers by thier index    \n",
    "    df=df.drop(test.index,axis=0)                        "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>age</th>\n",
       "      <th>sex</th>\n",
       "      <th>bmi</th>\n",
       "      <th>children</th>\n",
       "      <th>smoker</th>\n",
       "      <th>region</th>\n",
       "      <th>charges</th>\n",
       "      <th>weight_Condition</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>19</td>\n",
       "      <td>female</td>\n",
       "      <td>27.900</td>\n",
       "      <td>0</td>\n",
       "      <td>yes</td>\n",
       "      <td>southwest</td>\n",
       "      <td>16884.92400</td>\n",
       "      <td>Overweight</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>18</td>\n",
       "      <td>male</td>\n",
       "      <td>33.770</td>\n",
       "      <td>1</td>\n",
       "      <td>no</td>\n",
       "      <td>southeast</td>\n",
       "      <td>1725.55230</td>\n",
       "      <td>Obese</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>28</td>\n",
       "      <td>male</td>\n",
       "      <td>33.000</td>\n",
       "      <td>3</td>\n",
       "      <td>no</td>\n",
       "      <td>southeast</td>\n",
       "      <td>4449.46200</td>\n",
       "      <td>Obese</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>33</td>\n",
       "      <td>male</td>\n",
       "      <td>22.705</td>\n",
       "      <td>0</td>\n",
       "      <td>no</td>\n",
       "      <td>northwest</td>\n",
       "      <td>21984.47061</td>\n",
       "      <td>Normal</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>32</td>\n",
       "      <td>male</td>\n",
       "      <td>28.880</td>\n",
       "      <td>0</td>\n",
       "      <td>no</td>\n",
       "      <td>northwest</td>\n",
       "      <td>3866.85520</td>\n",
       "      <td>Overweight</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   age     sex     bmi  children smoker     region      charges  \\\n",
       "0   19  female  27.900         0    yes  southwest  16884.92400   \n",
       "1   18    male  33.770         1     no  southeast   1725.55230   \n",
       "2   28    male  33.000         3     no  southeast   4449.46200   \n",
       "3   33    male  22.705         0     no  northwest  21984.47061   \n",
       "4   32    male  28.880         0     no  northwest   3866.85520   \n",
       "\n",
       "  weight_Condition  \n",
       "0       Overweight  \n",
       "1            Obese  \n",
       "2            Obese  \n",
       "3           Normal  \n",
       "4       Overweight  "
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# CHANGE BMI to a Catogory for data training\n",
    "\n",
    "#function that will change  bmi to a category\n",
    "def weightCondition(bmi):\n",
    "  if bmi<18.5:\n",
    "    return \"Underweight\"\n",
    "  elif (bmi>= 18.5)&(bmi< 24.986):\n",
    "    return \"Normal\"\n",
    "  elif (bmi >= 25) & (bmi < 29.926):\n",
    "    return \"Overweight\"\n",
    "  else:\n",
    "    return \"Obese\"\n",
    "df[\"weight_Condition\"]=[weightCondition(val) for val in df[\"bmi\"] ]\n",
    "df.head(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "#TRAINING DATA\n",
    "\n",
    "#get the features and target col\n",
    "Y=df.charges\n",
    "X=df.drop([\"charges\"],axis=1)\n",
    "#train test split  \n",
    "x_train,x_test,y_train,y_test=train_test_split( X,Y,test_size=0.2,random_state=42)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>age</th>\n",
       "      <th>sex</th>\n",
       "      <th>bmi</th>\n",
       "      <th>children</th>\n",
       "      <th>smoker</th>\n",
       "      <th>region</th>\n",
       "      <th>weight_Condition</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1284</th>\n",
       "      <td>61</td>\n",
       "      <td>male</td>\n",
       "      <td>36.30</td>\n",
       "      <td>1</td>\n",
       "      <td>yes</td>\n",
       "      <td>southwest</td>\n",
       "      <td>Obese</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1114</th>\n",
       "      <td>23</td>\n",
       "      <td>male</td>\n",
       "      <td>24.51</td>\n",
       "      <td>0</td>\n",
       "      <td>no</td>\n",
       "      <td>northeast</td>\n",
       "      <td>Normal</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>969</th>\n",
       "      <td>39</td>\n",
       "      <td>female</td>\n",
       "      <td>34.32</td>\n",
       "      <td>5</td>\n",
       "      <td>no</td>\n",
       "      <td>southeast</td>\n",
       "      <td>Obese</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>600</th>\n",
       "      <td>18</td>\n",
       "      <td>female</td>\n",
       "      <td>39.16</td>\n",
       "      <td>0</td>\n",
       "      <td>no</td>\n",
       "      <td>southeast</td>\n",
       "      <td>Obese</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>171</th>\n",
       "      <td>49</td>\n",
       "      <td>male</td>\n",
       "      <td>30.30</td>\n",
       "      <td>0</td>\n",
       "      <td>no</td>\n",
       "      <td>southwest</td>\n",
       "      <td>Obese</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      age     sex    bmi  children smoker     region weight_Condition\n",
       "1284   61    male  36.30         1    yes  southwest            Obese\n",
       "1114   23    male  24.51         0     no  northeast           Normal\n",
       "969    39  female  34.32         5     no  southeast            Obese\n",
       "600    18  female  39.16         0     no  southeast            Obese\n",
       "171    49    male  30.30         0     no  southwest            Obese"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_train.head()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>age</th>\n",
       "      <th>sex</th>\n",
       "      <th>bmi</th>\n",
       "      <th>children</th>\n",
       "      <th>smoker</th>\n",
       "      <th>region</th>\n",
       "      <th>weight_Condition</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>901</th>\n",
       "      <td>60</td>\n",
       "      <td>male</td>\n",
       "      <td>40.92</td>\n",
       "      <td>0</td>\n",
       "      <td>yes</td>\n",
       "      <td>southeast</td>\n",
       "      <td>Obese</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1066</th>\n",
       "      <td>48</td>\n",
       "      <td>male</td>\n",
       "      <td>37.29</td>\n",
       "      <td>2</td>\n",
       "      <td>no</td>\n",
       "      <td>southeast</td>\n",
       "      <td>Obese</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1255</th>\n",
       "      <td>42</td>\n",
       "      <td>female</td>\n",
       "      <td>37.90</td>\n",
       "      <td>0</td>\n",
       "      <td>no</td>\n",
       "      <td>southwest</td>\n",
       "      <td>Obese</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>299</th>\n",
       "      <td>48</td>\n",
       "      <td>female</td>\n",
       "      <td>28.88</td>\n",
       "      <td>1</td>\n",
       "      <td>no</td>\n",
       "      <td>northwest</td>\n",
       "      <td>Overweight</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>238</th>\n",
       "      <td>19</td>\n",
       "      <td>male</td>\n",
       "      <td>29.07</td>\n",
       "      <td>0</td>\n",
       "      <td>yes</td>\n",
       "      <td>northwest</td>\n",
       "      <td>Overweight</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      age     sex    bmi  children smoker     region weight_Condition\n",
       "901    60    male  40.92         0    yes  southeast            Obese\n",
       "1066   48    male  37.29         2     no  southeast            Obese\n",
       "1255   42  female  37.90         0     no  southwest            Obese\n",
       "299    48  female  28.88         1     no  northwest       Overweight\n",
       "238    19    male  29.07         0    yes  northwest       Overweight"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_test.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1284    47403.8800\n",
       "1114     2396.0959\n",
       "969      8596.8278\n",
       "600      1633.0444\n",
       "171      8116.6800\n",
       "Name: charges, dtype: float64"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_train.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['Obese', 'Normal', 'Overweight', 'Underweight'], dtype=object)"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Checking for types of Weight Condition\n",
    "x_train[\"weight_Condition\"].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>0</th>\n",
       "      <th>1</th>\n",
       "      <th>2</th>\n",
       "      <th>3</th>\n",
       "      <th>4</th>\n",
       "      <th>5</th>\n",
       "      <th>6</th>\n",
       "      <th>7</th>\n",
       "      <th>8</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.979571</td>\n",
       "      <td>1.995322</td>\n",
       "      <td>-0.569406</td>\n",
       "      <td>-0.617113</td>\n",
       "      <td>1.751771</td>\n",
       "      <td>0.843691</td>\n",
       "      <td>1.554508</td>\n",
       "      <td>0.960143</td>\n",
       "      <td>-0.092229</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.979571</td>\n",
       "      <td>-0.501172</td>\n",
       "      <td>-0.569406</td>\n",
       "      <td>-0.617113</td>\n",
       "      <td>-0.570851</td>\n",
       "      <td>-1.647651</td>\n",
       "      <td>-1.154812</td>\n",
       "      <td>-1.019125</td>\n",
       "      <td>-0.918415</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>-1.020855</td>\n",
       "      <td>-0.501172</td>\n",
       "      <td>-0.569406</td>\n",
       "      <td>1.620448</td>\n",
       "      <td>-0.570851</td>\n",
       "      <td>0.843691</td>\n",
       "      <td>-0.014046</td>\n",
       "      <td>0.627747</td>\n",
       "      <td>3.212513</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>-1.020855</td>\n",
       "      <td>-0.501172</td>\n",
       "      <td>-0.569406</td>\n",
       "      <td>1.620448</td>\n",
       "      <td>-0.570851</td>\n",
       "      <td>0.843691</td>\n",
       "      <td>-1.511301</td>\n",
       "      <td>1.440271</td>\n",
       "      <td>-0.918415</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0.979571</td>\n",
       "      <td>-0.501172</td>\n",
       "      <td>-0.569406</td>\n",
       "      <td>-0.617113</td>\n",
       "      <td>1.751771</td>\n",
       "      <td>0.843691</td>\n",
       "      <td>0.698933</td>\n",
       "      <td>-0.047118</td>\n",
       "      <td>-0.918415</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          0         1         2         3         4         5         6  \\\n",
       "0  0.979571  1.995322 -0.569406 -0.617113  1.751771  0.843691  1.554508   \n",
       "1  0.979571 -0.501172 -0.569406 -0.617113 -0.570851 -1.647651 -1.154812   \n",
       "2 -1.020855 -0.501172 -0.569406  1.620448 -0.570851  0.843691 -0.014046   \n",
       "3 -1.020855 -0.501172 -0.569406  1.620448 -0.570851  0.843691 -1.511301   \n",
       "4  0.979571 -0.501172 -0.569406 -0.617113  1.751771  0.843691  0.698933   \n",
       "\n",
       "          7         8  \n",
       "0  0.960143 -0.092229  \n",
       "1 -1.019125 -0.918415  \n",
       "2  0.627747  3.212513  \n",
       "3  1.440271 -0.918415  \n",
       "4 -0.047118 -0.918415  "
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Machine learning algorithms cannot work with categorical data directly.\n",
    "\n",
    "#Categorical data must be converted to numbers.\n",
    "\n",
    "#pipe1 contain 2 encoder ,one hot encoder and ordinal encoder\n",
    "#one hot encoder includes sex,smoker,region, weight_condition\n",
    "#ordinal encode the weight_condition col because we arrange the order on this col\n",
    "#pipe2 just scale all the columns \n",
    "pipe1=ColumnTransformer(transformers=[(\"OHE\",OneHotEncoder(sparse=False,drop=\"first\"),\n",
    "                                       [\"sex\",\"smoker\",\"region\"]),\n",
    "                                     (\"ordinal\",OrdinalEncoder(categories=[['Underweight','Normal','Overweight','Obese']]),\n",
    "                                      [\"weight_Condition\"])]\n",
    "                        ,remainder=\"passthrough\")\n",
    "pipe2=ColumnTransformer(transformers=[(\"scaling\",StandardScaler(),[0,1,2,3,4,5,6,7,8])],\n",
    "                        remainder=\"passthrough\")\n",
    "\n",
    "pipe=Pipeline([(\"pipe1\",pipe1),(\"pipe2\",pipe2)])\n",
    "x_train=pd.DataFrame(pipe.fit_transform(x_train))\n",
    "x_test=pd.DataFrame(pipe.transform(x_test))\n",
    "x_train.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([29602.29516844])"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# LOGIC TO TEST NEW DATA ROW\n",
    "reg = LinearRegression().fit(x_train, y_train)\n",
    "\n",
    "response = [21, 'male', 42.1, 1, 'yes', 'southeast']\n",
    "response_df = pd.DataFrame([response], columns = ['age', 'sex', 'bmi', 'children', 'smoker', 'region'])\n",
    "\n",
    "response_df[\"weight_Condition\"]=[weightCondition(val) for val in response_df['bmi']]\n",
    "\n",
    "\n",
    "encoded_response_df = pd.DataFrame(pipe.transform(response_df))\n",
    "y_pred = reg.predict(encoded_response_df) \n",
    "\n",
    "y_pred\n",
    "        "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
