import pickle
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, ifft
import logging
import math

def sim(hz, tau, t0, tmax, sr):
    """
    sim = cos(2*pi*hz * t) * exp((t-t0) / tau) * u(t-t0)
    """
    om = 2 * np.pi * hz
    sim = [np.cos(om * i / sr) for i in range(tmax * sr)]
    sim = np.array(sim)
    for i in range(len(sim)):
        if i / sr < t0:
            sim[i] = 0
        else:
            sim[i] *= np.exp(-(i / sr - t0) / tau)
    return sim

def calc_note_amplitude(fid,l,hz):
    d = {}
    o = math.log(hz[-1]/hz[-2])
    for x,y in zip(hz,l):
        t = fid.hz2scale(x)

        d[t] = d[t] + o*y*y if t in d else o*y*y
        # d[t] = max(d[t],y) if t in d else y

    return [d[t] if t in d else 0 for t in range(0,128)]

class note_finder(object):
    '''
    bijection between frequency and note
    '''
    scale69 = 440
    scale = []
    def __init__(self):
        self.init()

    def init(self):
        for i in range(128):
            self.scale.append(self.scale69/32 * math.pow(2,(i-9.0)/12))
        # for _,hz in enumerate(scale):
            # print("{} : {:2f}".format(_,hz))

    def hz2scale(self,hz):
        if hz > self.scale[-1]:
            logging.error("Your frequency is {:4f}. It's too big!".format(hz))
            exit(0)
        o = np.searchsorted(self.scale,hz)
        diff1, diff2 = abs(self.scale[o] - mel(hz)), abs(self.scale[o-1] - mel(hz)) #
        if diff2 < diff1:
            return o-1
        else :
            return o

def sfft(x, sr, dt, max_hz=2000):
    o = int(sr * dt)
    c = [x[i:i + o] for i in range(0, len(x), o)]
    c.pop()

    fft_c = [np.array(abs(fft(_)[:int(max_hz / sr * o)])) for _ in c]
    # fft_c[time,hz]
    fft_c = np.array(fft_c)
    fft_c = np.transpose(fft_c)
    # print(fft_c.shape)
    time = np.linspace(0, len(x) / sr, len(c))
    hz = np.linspace(0, fft_c.shape[0] * sr / o, fft_c.shape[0])
    return time, hz, fft_c

def improve_stft(x, sr, dt, time, hz, fft_c):
    # fft_C is [hz * time]
    lim = np.max(fft_c) * 0.3
    eps = lim / 10
    #sharpen
    for i in range(len(fft_c)):
        for j in range(len(fft_c[0])):
            if fft_c[i, j] < lim:
                fft_c[i, j] = 0
    # note2lst
    eps = 1e-5
    results = []
    for i in range(fft_c.shape[0]):
        l, r = 0, 0
        for j in range(fft_c.shape[1]):
            if fft_c[i, j] > 0:
                l, r = j if l == 0 else l, j
            else:
                if l != 0:
                    results.append([i, l, r])
                    l, r = 0, 0
    # del same note
    ppop = []
    for i in range(len(results)):
        for j in range(i, len(results)):
            if i == j:
                continue
            if abs(results[i][0] - results[j][0]) < 3 and \
                    results[i][1] == results[j][1] and \
                            results[i][2] == results[j][2]:
                if j not in ppop:
                    ppop.append(j)
    ppop.sort(reverse=True)
    for it in ppop:
        results.pop(it)
    # result = [[a, b, c], ] where hz[a] means hz, time[b] and time[c] means start and end time of note
    for it in results:
        print(f'{it[0]}, {it[1]:.1f}, {it[2]:.1f}')
    input('wait')
    
    # find next&last same pitch

    for i in range(len(results)):
        start, end = 0, len(fft_c[0])
        for j in range(len(results)):
            if i == j or results[i][0] != results[j][0]: 
                continue
            if results[i][2] < results[j][1]:
                end = min(end, results[j][1])
            if results[i][1] > results[j][2]:
                start = max(start, results[j][2])

        print(f'results[0] {results[0]}')
        print(f'hz {hz[results[0][0]]}')
        print(f'start {start}\nend {end}')
        # slice and fft
        slice_fft =np.zeros(sr * dt)
        for i in range(len(x)):
            if start<x[i]<end:
                slice_fft[i]=x[i]
        slice_fft = fft(slice_fft)
        print(f'fftc_shape {len(fft_c)}, {len(fft_c[0])}')
        print(len(slice_fft), len(x), len(hz), len(time))
        fq = np.linspace(0, sr, len(x))
        plt.figure()
        plt.plot(fq[2: len(fq) // 2], np.abs(slice_fft[2: len(fq) //2]))
        plt.show()

        la = fft(slice_fft)
        la =la/np.max(la)
        for j in range(len(fft_c)):
            fft_c[i, j]*=la[j]
    # replace original fft_c

def draw_sfft(title, time, hz, fft_c, color=15, save=False, show=False):
    plt.contourf(time, hz, fft_c, color)
    plt.xlabel('time-axis')
    plt.ylabel('note-axis')
    plt.title(title.split('/')[-1])
    plt.colorbar()
    if save:
        plt.savefig(title.split('/')[-1].split('.')[0] + ".png")
    if show:
        plt.show()

def mel(f):
    '''
    https://blog.csdn.net/chumingqian/article/details/124950613
    '''
    return 2595 * np.log10(1 + f / 700)

def kill_harmonic_q(sc_lst, u1=1, u2=1):
    lim = max(0.75, np.max(sc_lst) / 23 * 0.75)
    ret = np.zeros_like(sc_lst)
    for i in range(sc_lst.shape[0]):
        flag = 0
        for j in range(sc_lst.shape[1]):
            flattened = np.log(sc_lst[i, j] + 1) #
            if sc_lst[i, j] > lim:
                ret[i, j] = flattened #
                flag = 1
            elif j < 58 and sc_lst[i, j] > u1 * lim:
                ret[i, j] = flattened
                flag = 1
            elif j < 55 and sc_lst[i, j] > u2 * lim and flag == 0:
                ret[i, j] = flattened
                flag = 1
    return np.transpose(ret)

def kill_harmonic(sc_lst, u1=0.8, u2=0.4):
    # sc_lst = [time, note]
    mm = np.max(sc_lst) #
    ret = np.zeros_like(sc_lst)
    for i in range(sc_lst.shape[0]):
        lim = max(mm * 0.07, np.percentile(sc_lst[i, :], q=10)) #
        flag = 0
        for j in range(sc_lst.shape[1]):
            flattened = np.log(sc_lst[i, j] + 1)  #
            if sc_lst[i, j] > lim:
                ret[i, j] = flattened  #
                flag = 1
                if j + 12 < sc_lst.shape[1]:
                    if sc_lst[i, j+12] >= lim * u1:
                        ret[i, j+12] = np.log(sc_lst[i, j+12] + 1)
                    else:
                        ret[i, j+12] = 0
                if j + 24 < sc_lst.shape[1]:
                    if sc_lst[i, j+24] >= lim * u2:
                        ret[i, j+24] = np.log(sc_lst[i, j+24] + 1)
                    else:
                        ret[i, j+24] = 0


    return np.transpose(ret)

def adjust_amp(hz, fft_c):
    '''
    a poly regression
    '''
    coef = [0.00000000e+00, -7.68673195e+03, 2.71093952e+04, -4.11494671e+04,
            3.48809954e+04, -1.80799373e+04, 5.87343617e+03, -1.16855713e+03,
            1.30256992e+02, -6.23211309e+00]
    intercept = 165.31216527

    for i in range(len(fft_c)):
        if hz[i] >= 20:
            ans, xx, lghz = intercept, 1, np.log10(hz[i])
            for k in range(len(coef)):
                ans += xx * coef[k]
                xx *= lghz
            for j in range(len(fft_c[0])):
                fft_c[i, j] *= (100 / ans) #
    return fft_c

def note_time_2_lst(ret, dt):
    eps = 1e-5
    results = []
    for i in range(ret.shape[0]):
        l, r = 0, 0
        for j in range(ret.shape[1]):
            if ret[i, j] > 0:
                l, r = j if l == 0 else l, j
            else:
                if l != 0:
                    results.append([i, l * dt, r * dt])
                    l, r = 0, 0

    return results

def main(File, dt=0.1, max_hz=2000):
    if File.split('.')[-1] == 'lst': 
        with open(File, "rb") as f:
            x, sr = pickle.load(f)
    else:
        import librosa
        x, sr = librosa.load(File, sr=44100)

    # x = x[0: len(x) // 2] #

    fid = note_finder()
    time, hz, fft_c = sfft(x, sr, dt, max_hz)
    # improve_stft(x, sr, dt, time, hz, fft_c)
    #draw_sfft(File, time, hz, fft_c, save=True)
    #input('no adjust')
    fft_c = adjust_amp(hz, fft_c)
    #draw_sfft(File, time, hz, fft_c, save=True)
    #input('after adjust')
    sc_lst = []
    for i in range(fft_c.shape[1]):
        sc_lst.append(np.array(calc_note_amplitude(fid, fft_c[:, i], hz)))
    note = range(128)

    sc_lst = np.array(sc_lst)
    ret = kill_harmonic(sc_lst)
    # ret = np.transpose(sc_lst)
    results = note_time_2_lst(ret, dt)
    return time, note, ret, results


if __name__ == '__main__':
    # path = '/storage/emulated/0/Android/data/org.qpython.qpy/files/05311402/'
    path = './data/'
    File = 'challenge-1.mp3'
    dt = 0.1
    max_hz = 2000
    time, note, tsrg, ret = main(path + File, dt, max_hz)

    for _ in ret:
        print("[{},{:1f},{:1f}]".format(_[0], _[1], _[2]))
    draw_sfft(File, time, note, tsrg, show=True)

