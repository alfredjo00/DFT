import seaborn as sns
from matplotlib import rc, rcParams


def set_params(text_size=25):
    sns.set('talk')

    font = {'family': 'STIXGeneral',
            'weight': 'normal',
            'size': text_size}
    rc('font', **font)
    rcParams['mathtext.fontset'] = 'stix'
    rcParams['font.family'] = 'STIXGeneral'