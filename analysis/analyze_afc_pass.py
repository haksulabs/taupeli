import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
import scipy.stats
import scipy.optimize
from scipy.special import logit, expit
from statsmodels.stats.proportion import proportion_confint
from numba import njit

rng = np.linspace(-4, 4)
def tau_estimate(duration, tau0, psi=1/2.0**2, m0=1.0, psi0=0.0):
    t = duration
    v0 = 0
    mean = m0*psi0 - psi0*t - psi*t**2 + psi*t*tau0
    mean /= t*psi + psi0

    subj_prec = psi0 + psi*t
    
    # TODO: Verify this is true
    #var = (t*psi + psi0**2*v0)/(psi0**2 + psi**2*t**2)
    var = 1/subj_prec

    subj_std = np.sqrt(1/subj_prec)
    return mean, np.sqrt(var), subj_std

dt = 1/100
@njit
def tau_estimate(duration, tau, p=1/2.0**2, taupow=0.0, est=1.0, p_est=0.0):
    t = 0
    est_var = 0.0
    speed = 1/tau
    position = 1
    p = p*dt
    while t < duration:
        t += dt
        tau -= dt
        position -= dt*speed
        obs = tau
        pred = est - dt
        ptau = p/(tau**taupow)
        K = ptau/(ptau + p_est)
        est = obs*K + (1 - K)*pred
        p_est += p
        
        est_var = K**2*(1/ptau) + (1 - K)**2*est_var
    

    return est, np.sqrt(est_var), np.sqrt(1/p_est)


#def correct_probr(intercept, beta_ttcdiff, beta_disappear):
#    def prob(ttcdiff, disappear):
#        x = intercept + 1.0/ttcdiff*beta_ttcdiff + beta_disappear*1/disappear
#        return psyfun(x)
#    return prob

# TODO: Slip probability
def correct_probr(psi, conf=2/3, disteffect=0.0):
    @np.vectorize
    def prob(maxttc, minttc, duration):
        m_min, s_min, subj_std_min = tau_estimate(duration, minttc, psi, disteffect)
        m_max, s_max, subj_std_max = tau_estimate(duration, maxttc, psi, disteffect)
        diff = m_min - m_max
        subj_diff_var = subj_std_max**2 + subj_std_min**2
        #subj_diff_prior = 1**2
        #subj_diff_var = 1/(1/subj_diff_var + 1/subj_diff_prior)
        #subj_diff_var = s_min**2 + s_max**2
        
        threshold = scipy.stats.norm.ppf(1 - conf, 0, np.sqrt(subj_diff_var))
        diff_var = s_min**2 + s_max**2
        p_correct = scipy.stats.norm.cdf(threshold, diff, np.sqrt(diff_var))
        p_wrong = 1 - scipy.stats.norm.cdf(-threshold, diff, np.sqrt(diff_var))
        #p = scipy.stats.norm.cdf(0, diff, np.sqrt(diff_var))
        p_pass = 1 - p_correct - p_wrong
        return p_correct, p_wrong, p_pass
    return prob


data = [json.loads(l) for l in sys.stdin]
data = pd.DataFrame.from_records(data)

data['ttc0'] = np.abs(1.0/data.v0)
data['ttc1'] = np.abs(1.0/data.v1)

data['maxttc'] = np.maximum(data.ttc0, data.ttc1)
data['minttc'] = np.minimum(data.ttc0, data.ttc1)
data['duration'] = (1 - data['disappear'])/(1/data.maxttc)

#data['maxspeed'] = np.maximum(data.v0, data.v1)

data = data.query("type == 'flash_nonforced'")
data = data[data.n_trials > 25]
data['correct'] = data.score > 0

def loss(p):
    prec = np.exp(p[0])
    conf = expit(p[1])
    correct, wrong, passed = correct_probr(prec, conf)(data.maxttc, data.minttc, data.duration)
    nonpassed = data.score != 0
    total = 0
    for i in range(len(correct)):
        p = {
                1: correct[i],
                -2: wrong[i],
                0: passed[i]
        }[data.score.iloc[i]]
        total += np.log(p)

    return -total


wtf = scipy.optimize.minimize(loss, [np.log(1/0.1**2), logit(2/3),], method='nelder-mead')

wtf.x[0] = np.exp(wtf.x[0])
wtf.x[1] = expit(wtf.x[1])
print(wtf)
std = np.sqrt(1/wtf.x[0])
print("Std per sec", std)
probr = correct_probr(*wtf.x)

minttc = 1.5
ttcdiffs = np.linspace(0.0, 0.5, 1000)
maxttc = ttcdiffs + minttc
disappears = np.linspace(0.4, 0.9, 4)

ttcdiffbins = np.linspace(0, 0.5, 7)

for i, disappear in enumerate(disappears[:-1]):
    s = disappear
    e = disappears[i+1]
    d = data[data.disappear.between(s, e)]
    for j in range(len(ttcdiffbins)-1):
        bd = d[d.ttcdiff.abs().between(*ttcdiffbins[[j, j+1]])]
        n_correct = (bd.score == 0).sum()
        x = ttcdiffbins[[j, j+1]].mean()
        x += disappear*0.01
        share = n_correct/len(bd)
        low, high = proportion_confint(n_correct, len(bd), method='beta')
        plt.errorbar(x, share, [[share-low], [high-share]], fmt='o', color=f"C{i}")
    duration = (1 - disappear)/(1/maxttc)
    probs = probr(maxttc, minttc, duration)
    plt.plot(ttcdiffs, probs[2], label=f"Disappear {disappear:.2f}", color=f"C{i}")
plt.ylabel("Share passed")
plt.xlabel("TTC difference")

#durations = np.linspace(0.1, 1.0, 5)
#for duration in durations:
#    plt.plot(ttcdiffs, probr(maxttc, minttc, duration), label=duration)


plt.legend()
plt.figure()

#plt.plot

plt.scatter(data.ttcdiff.abs(), data.disappear, c=data.score)
plt.colorbar()
plt.xlabel("Ttcdiff")
plt.ylabel("disappear")

plt.figure()
plt.scatter(data.ttcdiff.abs(), data.correct, alpha=0.2)

plt.show()
