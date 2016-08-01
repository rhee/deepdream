#

import sys,time

class nperf:

    maxcount = 0
    count = 0
    lastcount = 0
    tstart = 0
    interval = 5.0

    def __init__(self, interval = 5.0, maxcount = 0):
	self.interval = interval
	self.maxcount = maxcount
        self.count=self.lastcount=0
        self.tstart=time.time()

    def __call__(self, tag = 'nperf', callback = None):
        self.count += 1
        tlap = time.time()
        telapsed = tlap - self.tstart
        if telapsed >= self.interval:
            cps = (self.count - self.lastcount) / telapsed
            if self.maxcount > 0:
                eta = int((self.maxcount - self.count) / cps)
                eta_m = int(eta / 60)
                eta_s = eta % 60
                sys.stderr.write(
		    '%s count: %d/%d, elapsed %.1f speed %.1f steps/sec, eta %d:%02d' %
                    (tag, self.count, self.maxcount, telapsed, cps, eta_m, eta_s) + '\n')
            else:
                sys.stderr.write(
		    '%s count: %d, elapsed %.1f speed %.1f steps/sec' %
		    (tag, self.count, telapsed, cps) + '\n')

            self.lastcount=self.count
            self.tstart=tlap

            if not callback is None:
                callback(self.count, tlap)

if '__main__' == __name__:
    check = nperf(interval = 5.0, maxcount=250)
    for i in range(250):
        def callback(count, tlap):
            print 'callback: ', i, count, tlap
        check('hello', callback)
        time.sleep(0.1)

# Emacs:
# Local Variables:
# mode: python
# c-basic-offset: 4
# End:
# vim: sw=4 sts=4 ts=8 et ft=python:
