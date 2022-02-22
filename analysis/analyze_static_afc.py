import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats
import scipy.optimize

from statsmodels.stats.proportion import proportion_confint

data = map(json.loads, sys.stdin)
data = pd.DataFrame.from_records(data)

maxdiff = 0.1
min_dur = 1/30
max_dur = 0.3


data['minpos'] = np.minimum(data.x0.abs(), data.x1.abs())
data['absposdiff'] = data['posdiff'].abs()

def dist_estimate(distance, duration, psi=1.0, scale=1.0):
    t = duration
    value = -np.exp(-distance*scale)
    mean = value
    var = 1/(psi*t)

    return mean, np.sqrt(var)

def correct_probr(psi, scale=1.0, slip=0.0):
    @np.vectorize
    def prob(mindist, maxdist, duration):
        m_min, s_min = dist_estimate(mindist, duration, psi, scale)
        m_max, s_max = dist_estimate(maxdist, duration, psi, scale)
        diff = m_min - m_max
        diff_var = s_min**2 + s_max**2
        p = scipy.stats.norm.cdf(0, diff, np.sqrt(diff_var))
        
        # Probability of seeing
        p = (1 - slip)*p + 0.5*slip

        return p
    return prob

def loss(p):
    p = np.exp(p)
    probs = correct_probr(*p)(data.minpos, data.minpos + data.absposdiff, data.duration)
    lik = 0.0
    for i in range(len(probs)):
        if data.correct.iloc[i]:
            lik += np.log(probs[i])
        else:
            lik += np.log(1 - probs[i])
    return -lik

"""
fit = GaussianProcessRegressor().fit(data[['minpos', 'absposdiff']], data.correct)

for diff in [0, 0.05, 0.1]:
    posrng = np.linspace(0.2, 0.8, 1000)
    diffrng = np.zeros(len(posrng)) + diff

    rng = np.vstack((posrng, diffrng)).T
    print(fit.predict(rng))
    plt.plot(posrng, fit.predict(rng), label=f"Diff {diff}")

plt.legend()
"""

#logreg = smf.logit("correct ~ minpos*absposdiff*duration", data=data).fit()

wtf = scipy.optimize.minimize(loss, np.log([10000.0, 10.0]))
wtf.x = np.exp(wtf.x)
print(wtf)
probr = correct_probr(*wtf.x)
duration = data.duration.mean()
absposdiff = data.absposdiff.mean()

posbins = np.linspace(0.2, 0.8, 5)
diffbins = np.linspace(0, 0.1, 4)

posrng = np.linspace(0, 1, 1000)

for i, (s, e) in enumerate(zip(diffbins, diffbins[1:])):
    bd = data[data.absposdiff.between(s, e)]
    diff = np.mean([s, e])
    for s, e in zip(posbins, posbins[1:]):
        d = bd[bd.minpos.between(s, e)]
        x = np.mean([s, e])
        n_correct = np.sum(d.correct)
        share = np.mean(d.correct)
        low, high = proportion_confint(n_correct, len(d), method='beta')
        plt.errorbar(x, share, [[share-low], [high-share]], fmt=f'C{i}o')

    plt.plot(posrng, probr(posrng, posrng+diff, duration), f"C{i}", label=f"Position diff {diff:.2f}")

plt.xlabel("Distance from center")
plt.ylabel("Share correct")
plt.legend()
#plt.plot(data.minpos, data.correct, '.')
plt.show()
