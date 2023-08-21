"""
I/O routines for managing data associated with the 0vbb project
"""
import pandas as pd
import numpy as np
import gvar as gv
from tqdm import tqdm

def read_2pt(fname):
    """
    Two-point data from all configurations is concatenated together, with data
    from each configuration appearing in successive blocks.
    Each block consists of nt*nt total lines.
    The total number of lines in the file is nt*nt*nconfigs.
    """
    dataframe = pd.read_csv(fname, sep='\s+', header=None, names=['source','t','re','im'])
    nt = len(dataframe['t'].unique())
    nsrc = len(dataframe['source'].unique())
    if (len(dataframe) % (nt*nsrc) != 0):
        print("Unexpected line count:", len(dataframe))
    return dataframe

def avg_src_2pt(dataframe):
    """
    Average over the source times on each configuration.
    """
    nt = len(dataframe['t'].unique())
    nsrc = len(dataframe['source'].unique())
    if (len(dataframe) % (nt*nsrc)) != 0:
        print("Unexpected data size.")
    ncfg = len(dataframe) // (nt * nsrc)
    # Average over source times
    dataframe['config'] = np.repeat(np.arange(ncfg), nsrc*nt)
    dataframe = dataframe.groupby(['config','t'])['re'].mean().reset_index()
    # Reshape to (nconfigs, nt)
    return np.vstack(dataframe.groupby('t')['re'].apply(np.array).values).T


def get_2pt(fname, tcut, sign=+1):
    try:
        raw = read_2pt(fname)
    except FileNotFoundError:
        raise ValueError(f"{fname} not found")
    arr = avg_src_2pt(raw)
    arr = sign*arr[:, :tcut]
    dataframe = pd.DataFrame(arr, columns=np.arange(24))
    dataframe['config'] = dataframe.index
    return dataframe


def read_4pt(fname, nblock=6):
    """
    ty = location of first operator
    tx = location of second operator
    tz = location of sink operator
    """
    dataframe = pd.read_csv(fname, sep='\s+', header=None, names=['top1','top2','tsnk','re','im'])
    ntimes = len(dataframe.groupby(['top1','top2','tsnk']).size())
    if (len(dataframe) % ntimes) != 0:
        print("Unexpected data size.")
    dataframe['tsrc'] = 0
    # Create column with configuration number
    nlines = len(dataframe)
    if (nlines % (nblock * ntimes) == 0):
        nconfigs = nlines // (nblock*ntimes)
    else:
        print("Bad line count?")
    dataframe['config'] = np.repeat(np.arange(nconfigs), repeats=nblock*ntimes)
    # dataframe['block'] = np.tile(np.repeat(np.arange(nblock), repeats=ntimes), reps=nconfigs)
    # Average together data in blocks
    dataframe = dataframe.groupby(['top1','top2','tsnk','tsrc','config']).mean().reset_index()

    return dataframe

def compute_ratio(c2, df4):

    # Add useful time separations
    df4['t-'] = df4['tsrc'] - df4['top2']
    df4['t+'] = df4['tsnk'] - df4['top2']
    df4['v'] = df4['top1'] - df4['top2']
    df4['dt'] = df4['t+'] - df4['t-']
    df4['dt+'] = df4['t+'] - df4['v'].apply(lambda val: max(val, 0))
    df4['dt-'] = df4['v'].apply(lambda val: min(val, 0)) - df4['t-']
    df4['|v|'] = df4['v'].apply(np.abs)

    # Compute the ratio "C4/C2"
    # this is a slick way to get the ratio... divide C4 by C2[dt] where dt is accessed in the df4 DataFrame
    df4['ratio'] = df4['re'] / df4['dt'].apply(lambda dt: c2[dt])
    return df4


def correlate_ratio(df2, df4, ntimes=24):
    arr2 = df2[range(ntimes)].values    # all 2-point corr data as an array (n_cfgs, ntimes)
    tmp = df4.groupby(['tsrc', 'top1', 'top2','tsnk'])['re'].apply(np.array)
    arr4 = np.vstack(tmp.values).T

    # print(arr2.shape)
    # print(arr4.shape)
    # arr2 = (ncfgs, ntimes)
    # arr4 = (ncfgs, n_time_pairs), where n_time_pairs = number of unique idxs (t-, v, t+)

    # Correlate 2pt and 4pt results
    ds = gv.dataset.avg_data({'c2': arr2, 'c4': arr4})

    # print(ds['c2'])
    # print(ds['c4'])
    # ds['c2'] and ds['c4'] contain gvar objects with the correlator which have all the respective 
    # means and covariances stored there

    # Repackage 4pt to DataFrame
    df4 = tmp.keys().to_frame().reset_index(drop=True)
    df4['re'] = ds['c4']
    # correlated gvar dataset for c4 repackaged into df4['re']

    # Compute ratio, including correlations
    return compute_ratio(ds['c2'], df4)


def ensemble_average_ratio(df):

    # Compute ensemble average
    tmp = {key: subdf['ratio'].values for key, subdf in df.groupby(['dt+', 'dt-', 'v'])}
    ds = gv.dataset.avg_data(tmp)

    # Repackage to DataFrame
    def unpack(key, value):
        dtp, dtm, v = key
        return {'dt+': dtp, 'dt-': dtm, 'v': v, 'ratio': value}
    df = pd.DataFrame([unpack(key, value) for key, value in ds.items()])
    df['|v|'] = df['v'].apply(np.abs)
    return df


def avg_4pt(dataframe):
    # tmp = pandas Series, tmp[tsrc, top1, top2, tsnk] = np.array of all the real parts of each
    # correlator entry. Note that for a given t1, t2, t3, t4, len(tmp[t1, t2, t3, t4]) = number of configs
    tmp = dataframe.groupby(['tsrc', 'top1', 'top2','tsnk'])['re'].apply(np.array)

    # arr = numpy array of all  real parts of correlator data. Shape is (n_cfgs, n_time_idxs), so if you 
    # do arr[0], you'll get all the values of Re[C4] on configuration 0.
    arr = np.vstack(tmp.values).T

    # Compute ensemble average
    return pd.DataFrame(gv.dataset.avg_data(arr), index=tmp.index).\
        reset_index().\
        rename(columns={0:'corr'})
