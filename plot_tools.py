# Copyright IonQ Inc., all rights reserved
import numpy as np
import scipy.optimize as spo
import matplotlib.pyplot as plt
from pickle import load
from scipy.stats import norm, t

def z_alpha(confidence):
    '''2-sided quantile of a normal distribution'''
    alpha = 1-confidence
    conf_1_sided = 1 - alpha/2

    return norm.ppf(conf_1_sided)

def wilson(n, p, confidence):
    za = z_alpha(confidence)
    disc = np.sqrt(4*n*p*(1-p) + za**2)
    frac = 1/(1+za**2/n)

    swing = frac * za * disc / (2*n)
    midpt = frac * (p + za**2 / (2*n))

    return p - (midpt - swing), midpt + swing - p

def student(nvars, nparams, covariance, confidence=0.95):
    alpha = 1-confidence
    z = 1 - alpha/2
    tz = t.ppf(z, max(0, nvars - nparams))

    sigma = np.sqrt(np.diag(covariance)) * tz

    return sigma


def confidence_band(func, x, popt, pcov, confidence=0.95):                                                                                                                                                 
    """Compute upper and lower confidence bands for a fitted curve.
                                                                                                                                                                                                            
    Args:
        func: The fit function, called as func(x, *params).
        x: Array of x values to evaluate at.
        popt: Optimal parameters from curve_fit.
        pcov: Covariance matrix from curve_fit.
        confidence: Confidence level (default 0.95).

    Returns:
        lower, upper: Arrays of lower and upper bounds.
    """
    from scipy.stats import t as t_dist

    x = np.asarray(x)
    n_params = len(popt)

    # Compute the Jacobian: partial derivatives of func w.r.t. each parameter
    y0 = func(x, *popt)
    jac = np.zeros((len(x), n_params))
    eps = np.sqrt(np.finfo(float).eps)
    for i in range(n_params):
        dp = np.zeros(n_params)
        dp[i] = max(abs(popt[i]) * eps, eps)
        jac[:, i] = (func(x, *(popt + dp)) - y0) / dp[i]

    # Propagate uncertainty: variance at each x point
    # var(y) = J @ pcov @ J^T (diagonal elements)
    variance = np.sum((jac @ pcov) * jac, axis=1)
    sigma = np.sqrt(variance)

    # t-value for the desired confidence level (two-tailed)
    # Use a large dof if unknown; pass dof as n_data - n_params if available
    t_val = t_dist.ppf((1 + confidence) / 2, df=1e6)

    lower = y0 - t_val * sigma
    upper = y0 + t_val * sigma
    return lower, upper

def assemble_T_curve_data(alldata, fam, d_list, r_list, basis):
    outdata = []
    for d in d_list:
        plotd = []
        rmax = 0
        for r in r_list:
            rawsum = 0
            decsum = 0
            shosum = 0
            for key in alldata.keys():
                pfile = key.split('/')[-1]
                basename = pfile.split('_')[0]
                dstr = pfile.strip('ore/').split('_')[1]
                bstr = pfile.strip('ore/').split('_')[3].strip('.conf.js')
                rstr = pfile.strip('ore/').split('_')[2]

                stat = alldata[key]
                sho = len(stat)
                dis = int(dstr[1:])
                rou = int(rstr[1:])
                raw = np.sum([s[0] for s in stat])
                dec = np.sum([s[1] for s in stat])
                
                if dis==d and rou==r and basename==fam and bstr[0]==basis:
                    rawsum += raw
                    decsum += dec
                    shosum += sho
                    rmax = r
            plotd.append((rawsum / shosum, decsum / shosum, shosum, rmax))
        outdata.append(plotd)
    
    return outdata

def read_data(fname, code_fam, d_list, r_list, basis):
    handle = open(fname, 'rb')
    data = load(handle)
    handle.close()

    return assemble_T_curve_data(data, code_fam, d_list, r_list, basis)

def fit_decay_curve(data, confidence, shift=0, k=4):
    def fit_func(rounds, c, tau):
        exp = np.exp(-(rounds-shift)/tau)
        return c*(0.5 + 0.5*exp)**k

    fits = []
    for d in data:
        t_arr = np.array([tup[3] for tup in d])
        p_log = np.array([tup[1] for tup in d])
        shots = np.array([tup[2] for tup in d])
        wilson_lo, wilson_hi = wilson(shots, p_log, confidence)
        sigma = wilson_hi + wilson_lo

        sol = spo.curve_fit(fit_func, t_arr, (1-p_log), sigma=sigma)
        fits.append(sol)

    fits_raw = []
    for d in data:
        t_arr = np.array([tup[3] for tup in d])
        p_log = np.array([tup[0] for tup in d])
        shots = np.array([tup[2] for tup in d])
        wilson_lo, wilson_hi = wilson(shots, p_log, confidence)
        sigma = wilson_hi + wilson_lo

        sol = spo.curve_fit(fit_func, t_arr, (1-p_log), sigma=sigma)
        fits_raw.append(sol)
    
    return fits, fits_raw

class PlotHelper():
    def __init__(self):
        self.datasets = 0
        self.datalist = []
        self.datalbls = []
        self.datadims = (0,-1,-1)

    def add_dataset(self, data, lbls):
        '''ASSUMES each added data is a list of list,
        or a list of arrays of regular lengths, each
        of which contains a 4-tuple.'''
        if self.datadims[0] == 0: # First dataset
            self.datalist.append(data)
            self.datalbls.append(lbls)
            self.datadims = (1, len(data), len(data[0]))
        else:
            datashape = (len(data), len(data[0]))
            if self.datadims[1:3] != datashape:
                raise Exception("Incompatible shapes")
            else:
                self.datalist.append(data)
                self.datalbls.append(lbls)
                self.datadims = \
                    (self.datadims[0]+1, self.datadims[1], self.datadims[2])

    def parse_plot_allbar(self, data):
        '''Parse data, get average across all # of rounds.'''
        plot_raw = []
        plot_dec = []
        plot_shots = []
        for code in data:
            praw = 0
            plog = 0
            nshots = 0
            for r in code:
                raw, dec, shots, rounds = r
                praw += raw*shots/rounds
                plog += dec*shots/rounds
                nshots += shots
            plot_dec.append(plog / nshots)
            plot_raw.append(praw / nshots)
            plot_shots.append(nshots)
        
        return plot_raw, plot_dec, plot_shots

    def make_plot_allbar(self, x_offset, width, ybuff=0.05, ylabel='', xlabel='', tit=''):
        '''Bar chart with data from max # of rounds'''
        x = 1*np.arange(self.datadims[1]) + x_offset
        all_raw = []
        all_dec = []
        all_shots = []
        for data in self.datalist: # Loop through datasets e.g. unfiltered, or X vs Z
            raw, dec, shots = self.parse_plot_allbar(data)
            all_raw.append(raw)
            all_dec.append(dec)
            all_shots.append(shots)
        
        xlim = len(all_raw)*width-width/2
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Raw perf bar chart
        color = [(1.,0,0),(0.5,0,0.1),(1,1,0),(0.5,0.5,0)]
        for i,r in enumerate(all_raw):
            ax.bar(x-xlim+i*width, r, width, color=color[i],\
                label=f"Raw, ({self.datalbls[i]})")
            PlotHelper.add_plot_errorbar(ax,\
                x-xlim+i*width, r, all_shots[i])

        # Decoded perf bar chart
        color = [(0,0,1.),(0.1,0,0.5),(0,1,1),(0,0.5,0.5)]
        for i,r in enumerate(all_dec):
            ax.bar(x+i*width+width/2, r, width, color=color[i],\
                label=f"Decoded, ({self.datalbls[i]})")
            PlotHelper.add_plot_errorbar(ax,\
            x+i*width+width/2, r, all_shots[i])
        
        self.add_decorators_allbar(ax, x, all_raw, all_dec,\
            ybuff, ylabel, xlabel, tit)
        
        return fig, ax
    
    def add_decorators_allbar(self, ax, xarr, raw, dec, ybuffer, ylbl, xlbl, tit):
        '''Decorators for bar chart'''
        ax.set_xticks(xarr, [f"{v:1d}" for v in xarr])
        ymax = max(np.amax(raw), np.amax(dec))
        yticks = np.arange(0,ymax+ybuffer,0.1)
        ax.set_yticks(yticks, [f"{int(100*y):2d}" for y in yticks])
        ax.grid()

        ax.set_ylabel(ylbl)
        ax.set_xlabel(xlbl)
        ax.set_title(tit)
        
        ax.legend()
    
    @staticmethod
    def add_plot_errorbar(ax, xdat, ydat, shots, confidence=0.95, capsize=2, color='black', marker='d', alpha=1):
        '''Decorate with errorbars, assuming binomial spread'''
        yarr = np.array(ydat)
        # std = np.sqrt(yarr*(1-yarr)/shots) * 2
        err = wilson(shots, yarr, confidence)
        ax.errorbar(xdat, ydat, err,
            linestyle='', color=color,
            marker=marker, capsize=capsize,
            alpha=alpha
        )

    def add_loglog_errorbar(ax, xdat, ydat, shots, confidence=0.95, capsize=2, color='black', marker='d', alpha=1):
        '''Decorate with errorbars, assuming binomial spread'''
        yarr = np.array(ydat)
        # std = np.sqrt(yarr*(1-yarr)/shots) * 2
        err_linear = wilson(shots, yarr, confidence)
        hi = np.log10(ydat + err_linear[1]) - np.log10(ydat)
        lo = np.log10(ydat) - np.log10(ydat - err_linear[0])
        err_loglog = np.array([lo, hi])
        ax.errorbar(np.log10(xdat), np.log10(ydat), err_loglog,
            linestyle='', color=color,
            marker=marker, capsize=capsize,
            alpha=alpha
        )

    def make_T_curves(self, title, lbls):
        all_raw = []
        all_dec = []
        all_rnd = []
        all_shots = []
        for data in self.datalist: # Loop through datasets; X or Z etc.
            for code in data:
                all_raw.append(np.array([1-t[0] for t in code]))
                all_dec.append(np.array([1-t[1] for t in code]))
                all_rnd.append(np.arange(len(code))+1)
                all_shots.append(np.array([t[2] for t in code]))
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        colors = ['blue','red','green','black']
        for i,rnd in enumerate(all_rnd):
            ax.plot(rnd, all_raw[i], color=colors[i], linestyle='--')
            PlotHelper.add_plot_errorbar(ax, rnd, all_raw[i], all_shots[i])
        for i,rnd in enumerate(all_rnd):
            ax.plot(rnd, all_dec[i], color=colors[i], linestyle='-', label=lbls[i])
            PlotHelper.add_plot_errorbar(ax, rnd, all_dec[i], all_shots[i])
        
        ax.set_ylabel("Survival Probability\n(1 - Logical Error Rate)")
        ax.set_xlabel("# of Code Cycles")
        ax.set_title(title)
        ax.legend()
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.1f}%'))
        ax.text(0.02, 0.98, '(a)', transform=ax.transAxes,
                fontweight='bold', va='top', ha='left')

        return fig, ax

    def make_T_curves_band(self, title, lbls, colors, shift=0, confidence=0.95, ax=None, k=4, show_legend=False, show_grid=False, y_lim_low=0.):
        all_raw = []
        all_dec = []
        all_rnd = []
        all_shots = []
        for data in self.datalist:
            for code in data:
                all_raw.append(np.array([1-t[0] for t in code]))
                all_dec.append(np.array([1-t[1] for t in code]))
                all_rnd.append(np.arange(len(code))+1)
                all_shots.append(np.array([t[2] for t in code]))

        combined = list(zip(all_rnd, all_raw, all_dec, all_shots))
        fits, fits_raw = fit_decay_curve(
            [[(1-r, 1-d, s, rn) for r, d, s, rn in zip(raw, dec, shots, rnd)]
             for rnd, raw, dec, shots in combined],
            confidence=confidence,
            shift=shift,
            k=k
        )

        def fit_func(rounds, c, tau):
            exp = np.exp(-(rounds-shift)/tau)
            return c*(0.5 + 0.5*exp)**k

        if ax is None:
            fig, ax = plt.subplots(figsize=(6.5,4))
        else:
            fig = ax.get_figure()

        summary_stats = []
        for i, rnd in enumerate(all_rnd):
            rnd_fine = np.linspace(rnd[0], rnd[-1], 100)

            # Raw: dashed line + band
            popt_r, pcov_r = fits_raw[i]
            lo_r, hi_r = confidence_band(fit_func, rnd_fine, popt_r, pcov_r, confidence)
            ax.plot(rnd_fine, fit_func(rnd_fine, *popt_r), color=colors[i], linestyle='--', alpha=0.5)
            ax.fill_between(rnd_fine, lo_r, hi_r, color=colors[i], alpha=0.1)

            # Decoded: solid line + band
            popt_d, pcov_d = fits[i]

            sig = student(len(all_dec[i]), 2, pcov_d, confidence)
            tau = popt_d[1]
            delt= sig[1]

            pl = 0.5*(1-np.exp(-1/tau))
            hi = 0.5*(1-np.exp(-1/(tau-delt)))
            lo = 0.5*(1-np.exp(-1/(tau+delt)))

            # Calculate and save summary stats
            pl_spread = max(hi-pl, pl-lo)
            summary_stats.append((pl, pl_spread, tau, delt))
            
            lo_d, hi_d = confidence_band(fit_func, rnd_fine, popt_d, pcov_d, confidence)

            ax.plot(rnd_fine, fit_func(rnd_fine, *popt_d), color=colors[i], linestyle='-', label=lbls[i])
            ax.fill_between(rnd_fine, lo_d, hi_d, color=colors[i], alpha=0.2)

            # Data points (no lines)
            PlotHelper.add_plot_errorbar(ax, rnd, all_raw[i], all_shots[i],\
                confidence=confidence, marker='d', color=colors[i], alpha=0.5)
            PlotHelper.add_plot_errorbar(ax, rnd, all_dec[i], all_shots[i],\
                confidence=confidence, marker='o', color=colors[i])

        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y*100):2d}'))
        ax.set_ylabel("Survival Probability (%)")
        ax.set_xlabel("# of Syndrome Extraction Cycles")
        ax.set_ylim(y_lim_low,1)
        ax.set_title(title)
        # ax.text(-0.12, 1.08, '(b)', transform=ax.transAxes,
        #     fontweight='bold', fontsize=13, va='top', ha='left')
        if show_legend: ax.legend()
        if show_grid: ax.grid()

        return fig, ax, summary_stats

def print_summary_stats(stats):
    print(f"{'LER':23s} | {'Tau (SEC)':23s}") # Header
    for s in stats:
        pl, pl_spread, tau, delt = s
        print(f"{pl:5.4e}+/-{pl_spread:5.4e} | {tau:5.4e}+/-{delt:5.4e}")

def print_summary_stats_latex(statsX, statsZ):
    plog_string = ""
    for sx, sz in zip(statsX, statsZ):
        plx, plx_spread, taux, deltx = sx
        plz, plz_spread, tauz, deltz = sz

        s_plx = f"{100*plx:3.2f}\\pm{100*plx_spread:3.2f}"
        s_plz = f"{100*plz:3.2f}\\pm{100*plz_spread:3.2f}"
        s_taux =f"{taux:2.1f}\\pm{deltx:3.2f}"
        s_tauz =f"{tauz:2.1f}\\pm{deltz:3.2f}"

        # plog_string += f"& ${s_taux}$ & ${s_plx}$ & ${s_tauz}$ & ${s_plz}$ \\tabularnewline \n"
        plog_string += f"& ${s_plx}$ & ${s_taux}$ & ${s_plz}$ & ${s_tauz}$ \\tabularnewline \n"

    print(plog_string)

def print_lifetime_latex(lifetimes):
    tx, tz, sx, sz, tr, sr = lifetimes

    s_Tx = f"{tx:3.2f}\\pm{sx:3.2f}"
    s_Tz = f"{tz:3.2f}\\pm{sz:3.2f}"
    s_Tr = f"{tr:3.2f}\\pm{sr:3.2f}"
    print(f"& ${s_Tx}$ & ${s_Tz}$ & ${s_Tr}$ \\tabularnewline \n")


def get_ticks(maj0, maj1, skip=0, prune=0):
    tcks = []
    lbls = []
    for i in range(maj0, maj1):
        for j in range(1,10):
            val = np.log10(j*10**(i))
            if j==1:
                lab = eval("r'$10^{" + f"{i}"+"}$'")
            else:
                lab = ""
            lbls.append(lab)
            tcks.append(val)
    
    if prune<0 or skip>0:
        lbls = lbls[skip:prune]
        tcks = tcks[skip:prune]

    return lbls, tcks

def get_ticks_custom(maj0, maj1, skip=0, prune=0):
    tcks = []
    lbls = []
    for i in range(maj0, maj1):
        for j in range(1,10):
            val = np.log10(j*10**(i))
            if j in [4, 6, 8] and i==-1:
                # lab = r"$0.5\times$"
                lab = eval("r'$0." + f"{j}" + "\\times$'")
            elif j in [1,2] and i==0:
                lab = eval("r'$" + f"{j}" + "\\times$'")
            else:
                lab = ""
            lbls.append(lab)
            tcks.append(val)
    
    if prune<0 or skip>0:
        lbls = lbls[skip:prune]
        tcks = tcks[skip:prune]

    return lbls, tcks