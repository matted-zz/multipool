#!/usr/bin/python2.6
import matplotlib
# matplotlib.use("Agg")
import numpy, pylab, scipy.stats, sys
from collections import defaultdict

def load_table(fname, binsize=1000, verbose=False):
    fin = open(fname)
    temp = defaultdict(lambda : numpy.zeros(2))
    for line in fin:
        if line.startswith("#"): continue
        line = line.strip().split()
        line = map(float, line[:3])
        a, b = line[1:3]
        if(a<=0 or b<=0) and a+b>0: # we might miss really informative SNPs, but we're probably just missing fixated markers... so skip them
            if verbose: print >>sys.stderr, "skipping", line
            continue
        # if a+b <= 0: continue
        if False: # read downsampling
            a = scipy.stats.binom.rvs(a, 0.1)
            b = scipy.stats.binom.rvs(b, 0.1)
            if a<=0 or b<=0:
                continue
        temp[line[0] / binsize] += (a,b)
    fin.close()

    means = numpy.zeros(max(temp.keys())+1)
    counts = numpy.zeros(max(temp.keys())+1)
    variances = numpy.zeros(max(temp.keys())+1) + float("inf")
    for loc, (a,b) in temp.iteritems():
        p = 1.0*a/(a+b)
        means[loc] = a
        counts[loc] = a+b
        if a+b > 0:
            variances[loc] = p*(1.0-p) * (a+b)

    return means, variances, counts

def BINOM_PMF(k,n,p):
    if p == 0.0: return 1.0 if k == 0 else 0.0
    if p == 1.0: return 1.0 if k == n else 0.0
    return scipy.stats.binom.pmf(k,n,p)


def kalman(y, y_var, d, T, N, p, doLOD=True):
    mu = numpy.zeros(T)
    V = numpy.zeros(T)
    P = numpy.zeros(T)

    V_pstr = numpy.zeros(T)
    mu_pstr = numpy.zeros(T)

    c = numpy.ones(T)

    mu_initial = 0.5*N # initial parameters, assumed given
    V_initial = 0.25*N # ditto

    A = (1.0 - 2.0*p)
    C = 1.0 * d / N
    S = p*(1.0-p)*N

    K = V_initial*C[0]/(C[0]**2.0*V_initial + y_var[0])
    # print "first C:", c[0], "first K:", K, "Vinitial:", V_initial, "yvar[0]:", y_var[0]
    mu[0] = mu_initial + K*(y[0] - C[0]*mu_initial)
    V[0] = (1.0-K*C[0])*V_initial
    # P[0] = A**2.0*V_initial + S
    if y_var[0] != float("inf"):
        c[0] = scipy.stats.norm.pdf(y[0], C[0]*mu_initial, numpy.sqrt(C[0]**2.0*V_initial + y_var[0]))
    else:
        c[0] = 1.0

    # print P[0], V[0], V_initial

    # forward pass
    for i in xrange(1,T):
        # P[i] = A**2.0 * V[i] + S # how did this work?
        if i == 1:
            P[i-1] = A**2.0*V_initial + S
        else:
            P[i-1] = A**2.0*V[i-1] + S
        if y_var[i] == float("inf"): # no observation here
            K = 0
            c[i] = 1.0
        else:
            K = P[i-1]*C[i]/(C[i]**2.0*P[i-1]+y_var[i])
            c[i] = scipy.stats.norm.pdf(y[i], C[i]*(A*mu[i-1]+p*N), numpy.sqrt(C[i]**2.0*P[i-1] + y_var[i]))
            c[i] = max(c[i], 1e-300)
            assert(c[i] != 0)
        mu[i] = A * mu[i-1] + N*p + K * (y[i] - C[i]*(A*mu[i-1] + N*p))
        V[i] = (1.0-K*C[i])*P[i-1]

    V_pstr[-1] = V[-1]
    mu_pstr[-1] = mu[-1]

    logLik = numpy.sum(numpy.log(c))
    print "model log lik:", logLik

    # backward pass
    for i in xrange(T-2,-1,-1):
        # if doLOD and i % 100 == 0: print >>sys.stderr, i
        J = V[i]*A/P[i]
        mu_pstr[i] = mu[i] + J * (mu_pstr[i+1] - A*(mu[i]) - N*p)
        V_pstr[i] = V[i] + J**2.0 * (V_pstr[i+1] - P[i])

    return mu_pstr, V_pstr, logLik

def calcLODs(mu_pstr, V_pstr, T, N):
    LOD = numpy.zeros(T)
    dumbLOD = numpy.zeros(T)

    mu_initial = 0.5*N # initial parameters, assumed given
    V_initial = 0.25*N # ditto
    mu_MLE = numpy.zeros(T)
    
    # precomputed stuff.
    # do the integral over 0...1
    x = numpy.arange(0,1.001,0.005) # improve this?
    p_precomp = numpy.array([scipy.stats.norm.pdf(N*x, N*p_alt, numpy.sqrt(p_alt*(1.0-p_alt)*N)) for p_alt in x])
    reweighter = scipy.stats.norm.pdf(N*x, mu_initial, numpy.sqrt(V_initial)) + 1e-300
    p_precomp /= len(x)

    for i in xrange(T):
        if True:
            # reweighted posterior distribution by the inverse of the stationary distribution.
            temp = numpy.log(1e-300 + scipy.stats.norm.pdf(N*x, mu_pstr[i], numpy.sqrt(V_pstr[i]))) - numpy.log(reweighter)
            temp = numpy.exp(temp)
            # now, we calculate a bunch of integrals with grids by
            # multiplying by the rows of p_precomp.  each row
            # corresponds to a value of p' that we want to optimize
            # over.  pick the best one.
            allsums = numpy.dot(p_precomp, temp)
            p_alt = x[allsums.argmax()] * N
            mu_MLE[i] = p_alt
            LOD[i] = numpy.log10(N*(x[1]-x[0]) * allsums.max()) # numpy.sum(temp * scipy.stats.norm.pdf(N*x, p_alt, numpy.sqrt(p_alt/N*(1.0-p_alt/N)*N))))
            # LOD[i] = max(0.0, LOD[i])
            if y_var[i] == float("inf") or LOD[i] < 2.0:
                dumbLOD[i] = 0.0
            else:
                # dumbLOD[i] = numpy.log10((x[1]-x[0]) * numpy.sum(scipy.stats.norm.pdf(N*x, dumb_mu, numpy.sqrt(dumb_V)) 
                # * scipy.stats.norm.pdf(N*x, p_alt, numpy.sqrt(p_alt/N*(1.0-p_alt/N)*N))))
                delta = 10000/res/2
                tempA = numpy.sum(y[max(0,i-delta):min(T-1,i+delta+1)])
                tempB = numpy.sum(d[max(0,i-delta):min(T-1,i+delta+1)])
                p_alt = 1.0*tempA/tempB
                # dumbLOD[i] = 0.0
                if False:
                    dumbLOD[i] = (numpy.log10(numpy.sum([BINOM_PMF(tempA, tempB, 1.0*x/N) * BINOM_PMF(x, N, 1.0*p_alt) for x in numpy.arange(N+1)]))
                                  - numpy.log10(numpy.sum([BINOM_PMF(tempA, tempB, 1.0*x/N) * BINOM_PMF(x, N, 0.5) for x in numpy.arange(N+1)])))
                else:
                    # why did I do this?
                    # tempB = numpy.sum(d) / numpy.sum(d > 0) # replace with average coverage...
                    dumbLOD[i] = numpy.log10(scipy.stats.norm.pdf(p_alt, p_alt, numpy.sqrt(p_alt*(1.0-p_alt)*(1.0/tempB + 1.0/N)))) - numpy.log10(scipy.stats.norm.pdf(p_alt, 0.5, numpy.sqrt(0.25*(1.0/tempB + 1.0/N))))
                # print dumbLOD[i], altDumb, dumbLOD[i] - altDumb

            # print "dumbLOD:", dumbLOD[i], y[i], d[i]
            # assert(dumbLOD[i] >= 0.0)
        # alt_lik = numpy.sum(posterior * [fac(nA) for nA in xrange(N+1)] * [BINOM_PMF(nA, N*INFLATE, alt_prop) for nA in xrange(N*INFLATE+1)])
        # null_lik = sum([posterior[nA/INFLATE] * fac(nA) * scipy.stats.binom.pmf(nA, N*INFLATE, 0.5) for nA in xrange(N*INFLATE+1)])
    return LOD, dumbLOD, mu_MLE

def calcLODs_coupled(mu_pstr, V_pstr, mu_pstr2, V_pstr2, T, N):
    LOD = numpy.zeros(T)
    dumbLOD = numpy.zeros(T)

    mu_MLE = numpy.zeros(T)

    mu_initial = 0.5*N # initial parameters, assumed given
    V_initial = 0.25*N # ditto
    
    # precomputed stuff.
    # do the integral over 0...1
    x = numpy.arange(0,1.001,0.005) # improve this?
    p_precomp = numpy.array([scipy.stats.norm.pdf(N*x, N*p_alt, numpy.sqrt(p_alt*(1.0-p_alt)*N)) for p_alt in x])
    reweighter = scipy.stats.norm.pdf(N*x, mu_initial, numpy.sqrt(V_initial)) + 1e-300
    p_precomp /= len(x)

    for i in xrange(T):
        if True:
            temp = numpy.exp(numpy.log(1e-300+scipy.stats.norm.pdf(N*x, mu_pstr[i], numpy.sqrt(V_pstr[i]))) - numpy.log(reweighter))
            temp2 = numpy.exp(numpy.log(1e-300+scipy.stats.norm.pdf(N*x, mu_pstr2[i], numpy.sqrt(V_pstr2[i]))) - numpy.log(reweighter))
            allsums = numpy.dot(p_precomp, temp) * numpy.dot(p_precomp, temp2)
            p_alt = x[allsums.argmax()] * N
            mu_MLE[i] = p_alt
            LOD[i] = numpy.log10(N*(x[1]-x[0]) * allsums.max())

            dumbLOD[i] = 0.0
    return LOD, dumbLOD, mu_MLE

def calcLODs_multicoupled(mu_pstr_vec, V_pstr_vec, T, N):
    LOD = numpy.zeros(T)
    dumbLOD = numpy.zeros(T)

    mu_MLE = numpy.zeros(T)

    mu_initial = 0.5*N # initial parameters, assumed given
    V_initial = 0.25*N # ditto
    
    # precomputed stuff.
    # do the integral over 0...1
    x = numpy.arange(0,1.001,0.005) # improve this?
    p_precomp = numpy.array([scipy.stats.norm.pdf(N*x, N*p_alt, numpy.sqrt(p_alt*(1.0-p_alt)*N)) for p_alt in x])
    reweighter = scipy.stats.norm.pdf(N*x, mu_initial, numpy.sqrt(V_initial)) + 1e-300
    p_precomp /= len(x)

    for i in xrange(T):
        if True:
            # temp = numpy.exp(numpy.log(1e-100+scipy.stats.norm.pdf(N*x, mu_pstr[i], numpy.sqrt(V_pstr[i]))) - numpy.log(reweighter))
            # temp2 = numpy.exp(numpy.log(1e-100+scipy.stats.norm.pdf(N*x, mu_pstr2[i], numpy.sqrt(V_pstr2[i]))) - numpy.log(reweighter))
            # allsums = numpy.dot(p_precomp, temp) * numpy.dot(p_precomp, temp2)

            log_allsums = numpy.zeros(len(x))
            for mu_pstr, V_pstr in zip(mu_pstr_vec, V_pstr_vec):
                temp = numpy.exp(numpy.log(1e-300+scipy.stats.norm.pdf(N*x, mu_pstr[i], numpy.sqrt(V_pstr[i]))) - numpy.log(reweighter))
                log_allsums += numpy.log(numpy.dot(p_precomp, temp) + 1e-300)
            allsums = numpy.exp(log_allsums)

            p_alt = x[allsums.argmax()] * N
            mu_MLE[i] = p_alt
            LOD[i] = numpy.log10(N*(x[1]-x[0]) * allsums.max())

            dumbLOD[i] = 0.0
    return LOD, dumbLOD, mu_MLE

if __name__ == "__main__":
    N = int(sys.argv[1]) # individuals
    # p = 0.0002 # recombination fraction
    res = 100
    p = res/100.0/3300.0 # was 3000.0
    REVERSE = False # deprecated...
    SIMPLE = False
    MOREINFO = True # what's this?
    REPLICATES = True
    COMBINE_DATA = False

    y,y_var,d = load_table(sys.argv[2], res, True)
    if len(sys.argv) > 3:
        y2, y_var2, d2 = [], [], []
        for fname in sys.argv[2:]:
            temp1, temp2, temp3 = load_table(fname, res, True)
            y2.append(temp1)
            y_var2.append(temp2)
            d2.append(temp3)
    else:
        y2 = None

    print >>sys.stderr, "loaded %d informative reads" % sum(d)
    if REVERSE:
        y = y[::-1]
        y_var = y_var[::-1]
        d = d[::-1]

    if y2 is None:
        T = len(y) # observations (max time)
    else:
        T = min([len(temp) for temp in y2])
        y = y[:T]
        y_var = y_var[:T]
        d = d[:T]

        for i in xrange(len(y2)):
            y2[i] = y2[i][:T]
            y_var2[i] = y_var2[i][:T]
            d2[i] = d2[i][:T]

        if COMBINE_DATA: # this doesn't work any more (y2, etc. are now lists)
            d += d2
            y = (y+y2)
            y_var = (y_var + y_var2)
            y2 = None

    start, stop = 0,0 # T/2,T # 2*T/5, T # 2*T/5, T # 2*T/11, T
    y_var[start:stop] = float("inf")
    if y2 is not None:
        for i in xrange(len(y_var2)):
            y_var2[i][start:stop] = float("inf")
    d[start:stop] = 0
    if y2 is not None: 
        for i in xrange(len(d2)):
            d2[i][start:stop] = 0

    X = numpy.arange(0, T*res, res)

    if False:
        bestP = None
        bestLik = float("-inf")
        for p in numpy.arange(p/10.0, p*5.0, p/10.0):
            print p,
            mu_pstr, V_pstr, logLik = kalman(y, y_var, d, T, N, p)
            if logLik > bestLik:
                bestLik = logLik
                bestP = p
        print "best r:", bestP, "in cM:", 1.0*res/bestP/100.0
        p = bestP
    print "recombination fraction:", p, "in cM:", 1.0*res/p/100.0

    mu_pstr, V_pstr, logLik = kalman(y, y_var, d, T, N, p)
    if y2 is not None:
        mu_pstr2 = []
        V_pstr2 = []
        for curr_y2, curr_y_var2, curr_d2 in zip(y2, y_var2, d2):
            curr_mu_pstr2, curr_V_pstr2, ignored = kalman(curr_y2, curr_y_var2, curr_d2, T, N, p)
            pylab.plot(X, curr_y2/curr_d2, "+", alpha=0.6)
            pylab.plot(X, curr_mu_pstr2/N, lw=2)
            mu_pstr2.append(curr_mu_pstr2)
            V_pstr2.append(curr_V_pstr2)
            # print mu_pstr
            # print mu_pstr2

    LOD, dumbLOD, mu_MLE = calcLODs(mu_pstr, V_pstr, T, N)
    print "single LOD 1:", LOD
    if y2 is not None:
        # LOD2, dumbLOD2, mu_MLE2 = calcLODs(mu_pstr2, V_pstr2, T, N)
        # LOD3, dumbLOD3, mu_MLE3 = calcLODs_coupled(mu_pstr, V_pstr, mu_pstr2, V_pstr2, T, N)
        LOD3, dumbLOD3, mu_MLE3 = calcLODs_multicoupled(mu_pstr2, V_pstr2, T, N)

        if REPLICATES:
            LOD = LOD3
            dumbLOD3 = dumbLOD3
            mu_MLE = mu_MLE3
        else:
            LOD2, dumbLOD2, mu_MLE2 = calcLODs(mu_pstr2[1], V_pstr2[1], T, N)
            assert(len(mu_pstr2) == 2)
            LOD = LOD + LOD2 - LOD3
            print "coupled LOD:", LOD3
            print "single LOD 1:", LOD
            print "single LOD 2:", LOD2
            print "LOD vector:", LOD
            dumbLOD = dumbLOD+dumbLOD2 - dumbLOD3

    if False and not SIMPLE:
        pylab.subplot(211)
    if REVERSE:
        pylab.plot(X, y[::-1]/d[::-1], "+")
    else:
        pylab.plot(X, y/d, "r+", alpha=0.6)
    # pylab.plot(y_var, "bo")
    pylab.xlabel("bp (%d bp loci)" % res)
    pylab.ylabel("Allele frequency")

    # pylab.plot(y)
    # pylab.plot(y_var)
    # pylab.plot(mu, '--')
    if REVERSE:
        mu_pstr = mu_pstr[::-1]
        V_pstr = V_pstr[::-1]
        LOD = LOD[::-1]
        dumbLOD = dumbLOD[::-1]

    pylab.plot(X, mu_pstr/N, 'r', lw=2)
    # if MOREINFO: pylab.plot(X, mu_MLE/N, 'r--', lw=1)
    # if y2 is not None:
    #    if MOREINFO: pylab.plot(X, mu_MLE2/N, 'm--', lw=1)

    if T < 20:
        # print "filtered means (forward predictions):", mu
        print "posterior means:", mu_pstr
        # print "filtered variances (forward predictions):", V
        print "posterior variances:", V_pstr

    if y2 is not None:
        for val, alpha in [(0.025, 0.3), (0.005, 0.1), (0.0005, 0.05)]:
            for curr_mu_pstr2, curr_V_pstr2 in zip(mu_pstr2, V_pstr2):
                CI = scipy.stats.norm.isf(val, 0, numpy.sqrt(curr_V_pstr2))
                pylab.fill_between(X, (curr_mu_pstr2 - CI)/N, (curr_mu_pstr2 + CI)/N, alpha=alpha)
    else:
        for val, alpha in [(0.025, 0.3), (0.005, 0.1), (0.0005, 0.05)]:
            CI = scipy.stats.norm.isf(val, 0, numpy.sqrt(V_pstr))
            pylab.fill_between(X, (mu_pstr - CI)/N, (mu_pstr + CI)/N, color='r', alpha=alpha)


    # CI = scipy.stats.norm.isf(0.025, 0, numpy.sqrt(V))
    # pylab.fill_between(numpy.arange(T), mu - CI, mu + CI, color='y', alpha=0.3)
    # pylab.plot(V, '--')
    # pylab.plot(V_pstr, '--', lw=2)
    pylab.axhline(0.5, color='k', ls=':')

    temp = numpy.exp(LOD) / numpy.sum(numpy.exp(LOD))

    left = temp.argmax()
    right = left
    cumul = temp[left]
    while cumul < 0.50: # and (left >= 0 and right < T):
        if temp[left] >= temp[right] and left > 0 or right == T-1 and left > 0:
            left -= 1
            cumul += temp[left]
        elif right < T-1:
            right += 1
            cumul += temp[right]
        else:
            break
    print "other 50% credible interval spans", left*res, right*res, "length is:", (right-left)*res
            
    targStart = 204491 # RAD5
    targStop = 208600 # RAD5
    # targStart = 466631
    # targStop = 469723
    # targStart = 517350
    # targStop = 526628
    # targStart = 171070
    # targStop = 180309
    targStart = left*res
    targStop = right*res
    pylab.fill_between([targStart-res/2,targStop-res/2], 0, 1, color="k", alpha=0.2)
    
    pylab.axis([0,T*res,0,1])
    # pylab.axis([0,400000,0,1])
    if not SIMPLE:
        pylab.twinx()
        pylab.ylabel("LOD score")
        pylab.plot(X, LOD, 'g', lw=2)
        if False and MOREINFO: pylab.plot(X, dumbLOD/dumbLOD*dumbLOD, 'mo')

    cumul, mean = 0.0, 0.0
    left, right = None, None
    for i,val in enumerate(temp):
        cumul += val
        if cumul >= 0.05 and left is None:
            left = i-1
        if cumul >= 0.95 and right is None:
            right = i
        mean += val*i*res
    if left is None: left = 0
    if right is None: right = T

    fout = open(sys.argv[2] + ".results3.txt", 'w')

    print >>fout, mean
    print "90% credible interval spans", left*res, right*res, "length is:", (right-left)*res, "mean:", mean, "mode:", temp.argmax()*res
    if not SIMPLE:
        if False and MOREINFO: pylab.plot(X, temp / numpy.max(temp) * numpy.max(LOD), 'g')
        # pylab.plot(X, scipy.stats.norm.pdf(X, mean, stdev) / scipy.stats.norm.pdf(mean, mean, stdev) * numpy.max(LOD), 'g:')
    # print LOD

    if False and not SIMPLE:
        pylab.subplot(212)
    if N < 10000:
        posteriors = numpy.zeros((N,T))
        for c in xrange(T):
            posteriors[:,c] = scipy.stats.norm.pdf(numpy.arange(0,1.0,1.0/N), mu_pstr[c]/N, numpy.sqrt(V_pstr[c])/N)
            posteriors[:,c] /= numpy.sum(posteriors[:,c])
        if False and not SIMPLE:
            pylab.imshow((numpy.log10(posteriors)), interpolation="nearest", origin="lower", cmap=pylab.get_cmap("YlGnBu"))

    if True:
        maxLOD = LOD.max()
        maxIndex = LOD.argmax()
        print "max multi-locus LOD score at:", maxLOD, maxIndex*res
        print >>fout, maxIndex*res
        index = maxIndex
        while index >= 0 and LOD[index] > maxLOD-1.0:
            index -= 1
        left = index
        print "1-LOD interval from ", index*res,
        index = maxIndex
        while index < T and LOD[index] > maxLOD-1.0:
            index += 1
        print "to", index*res, "length is:", (index-left)*res

        # print "p_alternate at highest LOD score segment:", (means[maxIndex] + 1.0) / (N+2.0)
        D = 30
        try:
            print "sublocalized best location:", numpy.sum(numpy.arange(res*(maxIndex-D),res*(maxIndex+D+1),res)*numpy.exp(LOD[maxIndex-D:maxIndex+D+1])) / numpy.sum(numpy.exp(LOD[maxIndex-D:maxIndex+D+1]))
        except ValueError:
            pass

    print "single-locus test best LOD at:", dumbLOD.argmax()*res
    print >>fout, dumbLOD.argmax()*res
    left = dumbLOD.argmax()
    right = left
    while dumbLOD[left] != 0.0 and dumbLOD[left] > dumbLOD.max() - 1.0:
        left -= 1
    while dumbLOD[right] != 0.0 and dumbLOD[right] > dumbLOD.max() - 1.0:
        right += 1

    for i in xrange(dumbLOD.argmax(),-1,-1):
        if dumbLOD[i] >= dumbLOD.max() - 1.0 and i < left:
            left = i
    for i in xrange(dumbLOD.argmax(),T):
        if dumbLOD[i] >= dumbLOD.max() - 1.0 and i > right:
            right = i

    print "single locus test 1-LOD interval from %d to %d (%d bp total)" % (left*res, right*res, (right-left)*res)
    pylab.axis([0,T*res,0,LOD.max()+3])
    # pylab.axis([0,400000,0,LOD.max()+3])
    pylab.show()

    fout.close()    
