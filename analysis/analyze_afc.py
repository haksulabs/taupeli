import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
import scipy.stats
import scipy.optimize
from statsmodels.stats.proportion import proportion_confint
from numba import njit

def psyfun(x):
    p = 1/(1 + np.exp(-x))
    p *= 0.5
    p += 0.5
    return p

rng = np.linspace(-4, 4)
"""
dt = 1/100
def tau_estimate(dt, duration, tau0, psi=1/2.0**2, m0=1.0, psi0=1.0):
    psi = psi*dt
    n = duration/dt
    v0 = 0.0
    
    mean = -n**2*dt*psi - n*dt*psi0 + m0*psi0 + n*psi*tau0
    mean /= n*psi + psi0

    var = (n*psi + psi0**2*v0)/(n**2*psi**2 + 2*psi*psi0 + psi0**2)

    return mean, np.sqrt(var)
"""

def tau_estimate_not(duration, tau0, psi=1/2.0**2, m0=1.0, psi0=1/10**2):
    t = duration
    v0 = 0
    mean = m0*psi0 - psi0*t - psi*t**2 + psi*t*tau0
    mean /= t*psi + psi0

    var = (t*psi + psi0**2*v0)/(psi0**2 + psi**2*t**2)

    return mean, np.sqrt(var)

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
        #obs = np.log(tau)
        #pred = np.log(np.exp(est) - dt)
        obs = -np.exp(-position*taupow)
        pred = obs
        #pred = position
        ptau = p#/(position**taupow)
        K = ptau/(ptau + p_est)
        est = obs*K + (1 - K)*pred
        p_est += p
        
        est_var = K**2*(1/ptau) + (1 - K)**2*est_var
    

    return est, np.sqrt(est_var)


#def correct_probr(intercept, beta_ttcdiff, beta_disappear):
#    def prob(ttcdiff, disappear):
#        x = intercept + 1.0/ttcdiff*beta_ttcdiff + beta_disappear*1/disappear
#        return psyfun(x)
#    return prob

# TODO: Slip probability
def correct_probr(psi, slip=0.01, disteffect=0.0):
    @np.vectorize
    def prob(maxttc, minttc, duration):
        m_min, s_min = tau_estimate(duration, minttc, psi, disteffect)
        m_max, s_max = tau_estimate(duration, maxttc, psi, disteffect)
        diff = m_min - m_max
        diff_var = s_min**2 + s_max**2
        p = scipy.stats.norm.cdf(0, diff, np.sqrt(diff_var))
        
        # Probability of seeing
        p = (1 - slip)*p + 0.5*slip

        return p
    return prob


data = [json.loads(l) for l in sys.stdin]
data = pd.DataFrame.from_records(data)

data['ttc0'] = np.abs(1.0/data.v0)
data['ttc1'] = np.abs(1.0/data.v1)

data['maxttc'] = np.maximum(data.ttc0, data.ttc1)
data['minttc'] = np.minimum(data.ttc0, data.ttc1)
data['duration'] = (1 - data['disappear'])/(1/data.maxttc)

#data['maxspeed'] = np.maximum(data.v0, data.v1)

data = data.query("type == 'flash_forced'")
data = data[data.n_trials > 25]
data['correct'] = data.score > 0

def loss(p):
    p = np.exp(p)
    probs = correct_probr(*p)(data.maxttc, data.minttc, data.duration)
    lik = 0.0
    for i in range(len(probs)):
        if data.correct.iloc[i]:
            lik += np.log(probs[i])
        else:
            lik += np.log(1 - probs[i])
    #loss = probs*data.correct + (1-probs)*(~data.correct)
    #return -np.sum(np.log(loss))
    return -lik

wtf = scipy.optimize.minimize(loss, np.log([1/0.1**2, 1e-1, 1.0]), method='powell')

wtf.x = np.exp(wtf.x)
print(wtf)
std = np.sqrt(1/wtf.x[0])
print("Std per sec", std)
probr = correct_probr(*wtf.x)

#param = [1/0.086**2, 1e-1, 1.5]
#print(loss(np.log(param)))
#probr = correct_probr(*param)
minttc = 1.5
ttcdiffs = np.linspace(0.001, 0.5, 1000)
maxttc = ttcdiffs + minttc
disappears = np.linspace(0.4, 0.9, 4)

ttcdiffbins = np.linspace(0, 0.5, 7)

for i, disappear in enumerate(disappears[:-1]):
    s = disappear
    e = disappears[i+1]
    d = data[data.disappear.between(s, e)]
    for j in range(len(ttcdiffbins)-1):
        bd = d[d.ttcdiff.abs().between(*ttcdiffbins[[j, j+1]])]
        n_correct = (bd.score > 0).sum()
        x = ttcdiffbins[[j, j+1]].mean()
        x += disappear*0.01
        share = n_correct/len(bd)
        low, high = proportion_confint(n_correct, len(bd), method='beta')
        plt.errorbar(x, share, [[share-low], [high-share]], fmt='o', color=f"C{i}")
    plt.plot(ttcdiffs, probr(maxttc, minttc, (1 - disappear)/(1/maxttc)), label=f"Disappear {disappear:.2f}", color=f"C{i}")
plt.ylabel("Share correct")
plt.xlabel("TTC difference")

#durations = np.linspace(0.1, 1.0, 5)
#for duration in durations:
#    plt.plot(ttcdiffs, probr(maxttc, minttc, duration), label=duration)


plt.legend()

"""
plt.figure()

#plt.plot

plt.scatter(data.ttcdiff.abs(), data.disappear, c=data.score)
plt.colorbar()
plt.xlabel("Ttcdiff")
plt.ylabel("disappear")

plt.figure()
plt.scatter(data.ttcdiff.abs(), data.correct, alpha=0.2)
"""
plt.show()
